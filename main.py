import os
from datetime import datetime

import numpy as np
import pandas as pd
from transformers import BartForConditionalGeneration, AutoTokenizer, TrainingArguments, Trainer, EvalPrediction, \
    AutoModelForSeq2SeqLM
from datasets import Dataset
from sklearn.model_selection import train_test_split
from evaluate import load
import argparse
from dataclasses import dataclass


@dataclass
class BartDetox:
    model_name: str = 'SkolkovoInstitute/bart-base-detox'
    tokenizer: AutoTokenizer.from_pretrained = AutoTokenizer.from_pretrained(model_name)
    model: BartForConditionalGeneration.from_pretrained = BartForConditionalGeneration.from_pretrained(model_name)


@dataclass
class T5Formality:
    model_name: str = 'Isotonic/informal_to_formal'
    tokenizer: AutoTokenizer.from_pretrained = AutoTokenizer.from_pretrained(model_name)
    model: AutoModelForSeq2SeqLM.from_pretrained = AutoModelForSeq2SeqLM.from_pretrained(model_name)


@dataclass
class T5Detox:
    model_name: str = 's-nlp/t5-paranmt-detox'
    tokenizer: AutoTokenizer.from_pretrained = AutoTokenizer.from_pretrained(model_name)
    model: AutoModelForSeq2SeqLM.from_pretrained = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def get_compute_metrics(tokenizer: AutoTokenizer.from_pretrained):
    bert_score_metric = load('bertscore')
    rouge_metric = load('rouge')  # Wraps up several variations of ROUGE, including ROUGE-L.
    blue_metric = load('sacrebleu')  # SacreBLEU is a standard BLEU implementation that outputs the BLEU score.
    meteor_metric = load('meteor')

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = preds.argmax(-1)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = tokenizer.batch_decode(p.label_ids, skip_special_tokens=True)
        bert_score = bert_score_metric.compute(predictions=preds, references=references, lang='en')
        rouge = rouge_metric.compute(predictions=preds, references=references)
        blue = blue_metric.compute(predictions=preds, references=references)
        meteor = meteor_metric.compute(predictions=preds, references=references)

        return {bert_score_metric.name: np.array(bert_score['f1']).mean(),
                rouge_metric.name: rouge['rougeL'],
                blue_metric.name: blue['score'],
                meteor_metric.name: meteor['meteor']}

    return compute_metrics


def create_datasets(data_path: str, test_size: float) -> dict:
    assert os.path.exists(data_path)

    data = pd.read_csv(data_path)
    data.columns.values[0] = 'source'
    data.columns.values[1] = 'target'
    data.index.name = 'id'
    data.reset_index(inplace=True)

    data_without_duplicates = data.drop_duplicates(subset='source')
    duplicates = data[~data['id'].isin(data_without_duplicates['id'])]

    train_data_without_duplicates, test_data_without_duplicates = train_test_split(data_without_duplicates,
                                                                                   test_size=test_size, random_state=42)

    train_duplicates = duplicates[duplicates['source'].isin(train_data_without_duplicates['source'])]
    test_duplicates = duplicates[duplicates['source'].isin(test_data_without_duplicates['source'])]

    train_data = pd.concat([train_data_without_duplicates, train_duplicates])
    test_data = pd.concat([test_data_without_duplicates, test_duplicates])

    print(f'train_data_size: {len(train_data)}, test_data_size: {len(test_data)}')

    splits = {'train': train_data, 'test': test_data}
    datasets = {'train': None, 'test': None}

    for split_name, split_data in splits.items():
        # Create dataset
        dataset = Dataset.from_dict({'source': split_data['source'],
                                     'target': split_data['target']})
        dataset.set_format(type='torch', columns=['source', 'target'])
        datasets[split_name] = dataset

    return datasets


def eval_model(trainer: Trainer, model_name: str, is_zero_shot=False):
    trainer.model.eval()
    predictions = trainer.predict(trainer.eval_dataset)
    preds = predictions.predictions[0]
    preds = preds.argmax(-1)
    preds = trainer.tokenizer.batch_decode(preds, skip_special_tokens=True)

    output_file_name = f'{model_name}.txt'
    if is_zero_shot:
        output_file_name = f'{model_name}_zero.txt'

    output_path = os.path.join(trainer.args.output_dir, output_file_name)
    with open(output_path, 'w') as f:
        f.write(f'PREDICTED & TARGET\n\n')
        for i in range(len(preds)):
            src = trainer.eval_dataset[i]['source']
            target = trainer.eval_dataset[i]['target']
            f.write(f'src: {src}\npred: {preds[i]}\ntarget: {target}\n')
            f.write('-' * 100 + '\n')
        f.write('\n\n\nMETRICS\n\n')
        f.write(f'metrics: {predictions.metrics}')

    print(f'Output (metrics & predictions) saved to: {output_path}')


def train_model(model_obj, data_path: str, training_args: TrainingArguments, device: str = 'cpu',
                test_size: float = 0.1):
    assert device in ['cpu', 'cuda']

    tokenizer = model_obj.tokenizer
    model = model_obj.model

    def preprocess_dataset(dataset: Dataset):
        source_texts = dataset['source']
        model_inputs = tokenizer(source_texts, truncation=True, padding='max_length', max_length=128)

        target_texts = dataset['target']
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(target_texts, truncation=True, padding='max_length', max_length=128)

        model_inputs['labels'] = targets['input_ids']
        return model_inputs

    data = create_datasets(data_path, test_size)
    train_set = data['train'].map(preprocess_dataset, batched=True)
    test_set = data['test'].map(preprocess_dataset, batched=True)
    test_set = test_set.remove_columns('labels')

    # Train model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        compute_metrics=get_compute_metrics(tokenizer),
        tokenizer=tokenizer,
    )

    eval_model(trainer, model_obj.model_name.split('/')[1],
               is_zero_shot=True)  # Evaluate model in zero-shot settings

    trainer.train()
    return trainer


def main():
    t5_form_name = 't5-formality'
    t5_detox_name = 't5-detox'
    bart_detox_name = 'bart-detox'

    parser = argparse.ArgumentParser(description='BART Detox Training')
    parser.add_argument('--data-path', type=str, help='Path to the input data file')
    parser.add_argument('--output-dir', type=str, help='Path to the output directory', default='results')
    parser.add_argument('--device', type=str, help='Either cpu or cuda', default='cpu')
    parser.add_argument('--model-name', type=str, help='Name of the model to use',
                        choices=[bart_detox_name, t5_form_name, t5_detox_name])

    args = parser.parse_args()

    now = str(datetime.now()).replace(' ', '_').replace(':', '_').split('.')[0]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_dir = os.path.join(args.output_dir, now)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
    )

    model_obj = None
    if args.model_name == t5_form_name:
        model_obj = T5Formality()
    elif args.model_name == bart_detox_name:
        model_obj = BartDetox()
    elif args.model_name == t5_detox_name:
        model_obj = T5Detox()

    trainer = train_model(model_obj, args.data_path, training_args, device=args.device)
    eval_model(trainer, args.model_name)


if __name__ == '__main__':
    main()
