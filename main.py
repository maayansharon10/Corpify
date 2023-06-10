import os
from datetime import datetime

import pandas as pd
from transformers import BartForConditionalGeneration, AutoTokenizer, TrainingArguments, Trainer, EvalPrediction
from datasets import Dataset
from sklearn.model_selection import train_test_split
from evaluate import load
import argparse


def get_compute_metrics(tokenizer: AutoTokenizer.from_pretrained):
    # bert_score_metric = load("bertscore")
    rouge_metric = load('rouge')  # Wraps up several variations of ROUGE, including ROUGE-L.
    blue_metric = load('sacrebleu')  # SacreBLEU is a standard BLEU implementation that outputs the BLEU score.

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = preds.argmax(-1)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = tokenizer.batch_decode(p.label_ids, skip_special_tokens=True)
        # bert_score = bert_score_metric.compute(predictions=preds, references=references, lang='en')
        rouge = rouge_metric.compute(predictions=preds, references=references)
        blue = blue_metric.compute(predictions=preds, references=references)
        # return {bert_score_metric.name: bert_score['f1'].mean(), rouge_metric.name: rouge['rougeL'].mean(),
        #         blue_metric.name: blue['score']}
        return {blue_metric.name: blue['score'], rouge_metric.name: rouge['rougeL'].mean()}

    return compute_metrics


def create_dataset(data_path: str, tokenizer: AutoTokenizer.from_pretrained, test_size: float) -> dict:
    assert os.path.exists(data_path)

    data = pd.read_csv(data_path)
    data.columns.values[1] = 'source'
    data.columns.values[2] = 'target'
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
        # Tokenize data
        src_encoding = tokenizer(split_data['source'].tolist(), truncation=True, padding=True)
        target_encoding = tokenizer(split_data['target'].tolist(), truncation=True, padding=True)

        # Create dataset
        dataset = Dataset.from_dict({'input_ids': src_encoding['input_ids'],
                                     'attention_mask': src_encoding['attention_mask'],
                                     'decoder_input_ids': target_encoding['input_ids'],
                                     'labels': target_encoding['input_ids']})
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        datasets[split_name] = dataset

    return datasets


def train_bart_detox(data_path: str, training_args: TrainingArguments, device: str = 'cpu', test_size: float = 0.2):
    assert device in ['cpu', 'cuda']

    model_name = 'SkolkovoInstitute/bart-base-detox'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

    dataset = create_dataset(data_path, tokenizer, test_size)
    # Train model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=get_compute_metrics(tokenizer),
        tokenizer=tokenizer,
    )

    trainer.train()

    model.eval()
    predictions = trainer.predict(dataset['test'])
    preds = predictions.predictions[0]
    preds = preds.argmax(-1)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

    now = datetime.now()
    with open(os.path.join(training_args.output_dir, f'results_{now}.txt'), 'w') as f:
        for i in range(len(preds)):
            src = dataset['test'][i]['input_ids']
            src = tokenizer.decode(src, skip_special_tokens=True)
            f.write(f'src: {src}\npred: {preds[i]}\ntarget: {labels[i]}\n')
            f.write(f'metrics: {predictions.metrics}')


def main():
    parser = argparse.ArgumentParser(description="BART Detox Training")
    parser.add_argument("--data-path", type=str, help="Path to the input data file")
    parser.add_argument("--output-dir", type=str, help="Path to the output directory", default='results')
    parser.add_argument("--device", type=str, help="Either cpu or cuda", default='cpu')

    args = parser.parse_args()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
    )

    train_bart_detox(args.data_path, training_args, device=args.device)


if __name__ == '__main__':
    main()
