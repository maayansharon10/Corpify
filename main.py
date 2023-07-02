import os
from datetime import datetime

import numpy as np
import pandas as pd
from transformers import BartForConditionalGeneration, AutoTokenizer, TrainingArguments, Trainer, EvalPrediction, \
    Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, Seq2SeqTrainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from evaluate import load
import argparse
from abc import ABC, abstractmethod


def create_datasets(data_path: str, test_size: float) -> dict:
    assert os.path.exists(data_path)

    data = pd.read_csv(data_path)
    data.columns.values[0] = 'source'
    data.columns.values[1] = 'target'
    data.index.name = 'id'
    data.reset_index(inplace=True)

    # Duplication control
    max_dups = 5
    sorted_data = data.sort_values(['source', 'id'])

    occurrence_counter = {}

    # Remove duplicates while keeping track of occurrences
    unique_data = []
    for _, row in sorted_data.iterrows():
        source = row['source']
        if source not in occurrence_counter:
            occurrence_counter[source] = 1
            unique_data.append(row)
        else:
            if occurrence_counter[source] <= max_dups:
                occurrence_counter[source] += 1
                unique_data.append(row)

    data = pd.DataFrame(unique_data)

    # Leakage control:
    data_without_duplicates = data.drop_duplicates(subset='source')
    duplicates = data[~data['id'].isin(data_without_duplicates['id'])]

    train_data_without_duplicates, test_data_without_duplicates = train_test_split(data_without_duplicates,
                                                                                   test_size=test_size,
                                                                                   random_state=42)

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


class RephrasingModel(ABC):
    def __init__(self, model_name: str, device: str, data_path: str, test_size: float, output_dir: str,
                 max_input_length: int):
        assert device in ['cpu', 'cuda']
        assert os.path.exists(data_path)
        assert 0 < test_size < 1
        assert os.path.exists(output_dir)
        assert max_input_length > 0

        self.model_name: str = model_name
        self.device: str = device
        self.data_path: str = data_path
        self.test_size: float = test_size
        self.output_dir: str = output_dir
        self.max_input_length: int = max_input_length

    @abstractmethod
    def create_trainer(self):
        pass

    def decode_preds(self, p: EvalPrediction, tokenizer):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = preds.argmax(-1)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        return preds

    def compute_metrics(self, p: EvalPrediction, eval_dataset: Dataset, tokenizer):
        bert_score_metric = load('bertscore')
        rouge_metric = load('rouge')  # Wraps up several variations of ROUGE, including ROUGE-L.
        blue_metric = load('sacrebleu')  # SacreBLEU is a standard BLEU implementation that outputs the BLEU score.
        meteor_metric = load('meteor')

        preds = self.decode_preds(p, tokenizer)
        references = eval_dataset['target']
        bert_score = bert_score_metric.compute(predictions=preds, references=references, lang='en')
        rouge = rouge_metric.compute(predictions=preds, references=references)
        blue = blue_metric.compute(predictions=preds, references=references)
        meteor = meteor_metric.compute(predictions=preds, references=references)

        return {bert_score_metric.name: np.array(bert_score['f1']).mean(),
                rouge_metric.name: rouge['rougeL'],
                blue_metric.name: blue['score'],
                meteor_metric.name: meteor['meteor']}

    def get_data_preprocessing_func(self, tokenizer):
        def preprocess_dataset(dataset: Dataset):
            source_texts = dataset['source']
            model_inputs = tokenizer(source_texts, truncation=True, padding='max_length',
                                     max_length=self.max_input_length)

            target_texts = dataset['target']
            with tokenizer.as_target_tokenizer():
                targets = tokenizer(target_texts, truncation=True, padding='max_length',
                                    max_length=self.max_input_length)

            model_inputs['labels'] = targets['input_ids']
            return model_inputs

        return preprocess_dataset

    def train(self, trainer):
        self.evaluate(trainer, is_zero_shot=True)
        trainer.train()

    def evaluate(self, trainer, is_zero_shot=False):
        trainer.model.eval()
        p = trainer.predict(trainer.eval_dataset)
        custome_metrics = self.compute_metrics(p, trainer.eval_dataset, trainer.tokenizer)
        preds = self.decode_preds(p, trainer.tokenizer)

        model_name = self.model_name.split('/')[1]
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
            f.write(f'metrics: {p.metrics}\n')
            f.write(f'costume metrics: {custome_metrics}\n')

        print(f'Output (metrics & predictions) saved to: {output_path}')


class BartDetox(RephrasingModel):
    def __init__(self, device: str, data_path: str, test_size: float, output_dir: str, max_input_length: int):
        super().__init__('SkolkovoInstitute/bart-base-detox', device, data_path, test_size, output_dir,
                         max_input_length)

        self.trainer = self.create_trainer()

    def create_trainer(self):
        model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        data = create_datasets(self.data_path, self.test_size)
        train_set = data['train'].map(self.get_data_preprocessing_func(tokenizer), batched=True)
        test_set = data['test'].map(self.get_data_preprocessing_func(tokenizer), batched=True)
        test_set = test_set.remove_columns('labels')

        training_args = TrainingArguments(
            output_dir=self.output_dir,
        )

        # Train model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=test_set,
            tokenizer=tokenizer,
        )

        return trainer

    def train_bart_detox(self):
        super().train(self.trainer)

    def evaluate_bart_detox(self, is_zero_shot=False):
        super().evaluate(self.trainer, is_zero_shot)


class T5Model(RephrasingModel):
    def __init__(self, name: str, device: str, data_path: str, test_size: float, output_dir: str,
                 max_input_length: int):
        super().__init__(name, device, data_path, test_size, output_dir,
                         max_input_length)

        self.trainer = self.create_trainer()

    def decode_preds(self, p: EvalPrediction, tokenizer):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        return preds

    def create_trainer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        data = create_datasets(self.data_path, self.test_size)
        train_set = data['train'].map(self.get_data_preprocessing_func(tokenizer), batched=True)
        test_set = data['test'].map(self.get_data_preprocessing_func(tokenizer), batched=True)

        args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            fp16=True,
            predict_with_generate=True,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_set,
            eval_dataset=test_set,
            tokenizer=tokenizer,
        )

        return trainer

    def train_t5(self):
        super().train(self.trainer)

    def evaluate_t5(self, is_zero_shot=False):
        super().evaluate(self.trainer, is_zero_shot)


class T5Formality(T5Model):
    def __init__(self, device: str, data_path: str, test_size: float, output_dir: str, max_input_length: int):
        super().__init__('Isotonic/informal_to_formal', device, data_path, test_size, output_dir,
                         max_input_length)


class T5Detox(T5Model):
    def __init__(self, device: str, data_path: str, test_size: float, output_dir: str, max_input_length: int):
        super().__init__('s-nlp/t5-paranmt-detox', device, data_path, test_size, output_dir, max_input_length)

        self.trainer = self.create_trainer()


def main():
    bart_detox_name = 'bart-detox'
    t5_formality_name = 't5-formality'
    t5_detox_name = 't5-detox'

    parser = argparse.ArgumentParser(description='BART Detox Training')
    parser.add_argument('--data-path', type=str, help='Path to the input data file')
    parser.add_argument('--output-dir', type=str, help='Path to the output directory', default='results')
    parser.add_argument('--device', type=str, help='Either cpu or cuda', default='cpu')
    parser.add_argument('--model-name', type=str, help='Name of the model to use',
                        choices=[bart_detox_name, t5_formality_name, t5_detox_name])

    args = parser.parse_args()

    now = str(datetime.now()).replace(' ', '_').replace(':', '_').split('.')[0]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_dir = os.path.join(args.output_dir, now)
    os.makedirs(output_dir, exist_ok=True)

    if args.model_name == bart_detox_name:
        model = BartDetox(args.device, args.data_path, test_size=0.1, output_dir=output_dir, max_input_length=128)
        model.train_bart_detox()
        model.evaluate_bart_detox()
    elif args.model_name == t5_formality_name:
        model = T5Formality(args.device, args.data_path, test_size=0.1, output_dir=output_dir, max_input_length=128)
        model.train_t5()
        model.evaluate_t5()
    elif args.model_name == t5_detox_name:
        model = T5Detox(args.device, args.data_path, test_size=0.1, output_dir=output_dir, max_input_length=128)
        model.train_t5()
        model.evaluate_t5()


if __name__ == '__main__':
    main()
