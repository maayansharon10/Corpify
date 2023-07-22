import json
import os
from typing import Dict

import wandb
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

from transformers.integrations import WandbCallback


def split_data(data: pd.DataFrame, max_dups: int, eval_size: float) -> (pd.DataFrame, pd.DataFrame):
    # Duplication control
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
    diff_rows = sorted_data.compare(data)

    print(f'Number of duplicates: {len(data) - len(unique_data)}')
    if len(data) - len(unique_data) > 0:
        print(f'Duplicate rows:\n{diff_rows}')

    # Leakage control:
    data_without_duplicates = data.drop_duplicates(subset='source')
    duplicates = data[~data['id'].isin(data_without_duplicates['id'])]

    train_data_without_duplicates, test_data_without_duplicates = train_test_split(data_without_duplicates,
                                                                                   test_size=eval_size,
                                                                                   random_state=42)

    train_duplicates = duplicates[duplicates['source'].isin(train_data_without_duplicates['source'])]
    test_duplicates = duplicates[duplicates['source'].isin(test_data_without_duplicates['source'])]

    train_data = pd.concat([train_data_without_duplicates, train_duplicates])
    test_data = pd.concat([test_data_without_duplicates, test_duplicates])

    return train_data, test_data


def filter_bad_ascii(df: pd.DataFrame) -> pd.DataFrame:
    filtered_sentences = []
    for index, row in df.iterrows():
        source_sentence = row['regular']
        target_sentence = row['corp']

        # Check if any character in the source sentence is not within ASCII range
        if any(ord(char) > 127 for char in source_sentence) or any(ord(char) < 32 for char in source_sentence):
            continue  # Skip this sentence

        # Check if any character in the target sentence is not within ASCII range
        if any(ord(char) > 127 for char in target_sentence) or any(ord(char) < 32 for char in target_sentence):
            continue  # Skip this sentence

        # If both sentences passed the check, add them to the filtered list
        filtered_sentences.append((source_sentence, target_sentence))

    # Create a new dataframe from the filtered sentences
    filtered_df = pd.DataFrame(filtered_sentences, columns=['regular', 'corp'])

    diff = len(df) - len(filtered_df)
    diff_rows = df.compare(filtered_df)

    print(f'Filtered {diff} rows with bad ASCII characters')
    if diff > 0:
        print(f'Filtered rows:\n{diff_rows}')

    return filtered_df


def create_datasets(data_path: str, max_dups, eval_size: float) -> dict:
    assert os.path.exists(data_path)

    data = pd.read_csv(data_path)
    data = filter_bad_ascii(data)
    data.columns.values[0] = 'source'
    data.columns.values[1] = 'target'
    data.index.name = 'id'
    data.reset_index(inplace=True)
    train_data, test_data = split_data(data, max_dups, eval_size)
    val_data, test_data = split_data(test_data, max_dups, eval_size=0.5)

    print(f'train_data_size: {len(train_data)}, test_data_size: {len(test_data)}')

    splits = {'train': train_data, 'validate': val_data, 'test': test_data}
    datasets = {'train': None, 'validate': None, 'test': None}

    for split_name, split in splits.items():
        # Create dataset
        dataset = Dataset.from_dict({'source': split['source'],
                                     'target': split['target']})
        dataset.set_format(type='torch', columns=['source', 'target'])
        datasets[split_name] = dataset

    return datasets


class RephrasingModel(ABC):
    def __init__(self, model_name: str, device: str, data_path: str, pipeline_config_args: Dict, output_dir: str,
                 max_input_length: int, load_from_checkpoint: bool):
        assert device in ['cpu', 'cuda']
        assert os.path.exists(data_path)
        assert 0 < pipeline_config_args["eval_size"] < 1
        assert os.path.exists(output_dir)
        assert max_input_length > 0

        self.model_name: str = model_name
        self.device: str = device
        self.data_path: str = data_path
        self.pipeline_config_args: Dict = pipeline_config_args
        self.output_dir: str = output_dir
        self.max_input_length: int = max_input_length
        self.load_from_checkpoint: bool = load_from_checkpoint

    def init_wandb_run(self, name: str):
        wandb_config = {
            "max_input_length": self.max_input_length,
            "model_name": self.model_name,
            "max_dups": self.pipeline_config_args["max_dups"],
            "eval_size": self.pipeline_config_args["eval_size"],
        }

        os.environ["WANDB_PROJECT"] = self.pipeline_config_args["wandb_project"]
        wandb.init(
            project=self.pipeline_config_args["wandb_project"],
            config=wandb_config,
            name=name
        )

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

    def save_best_checkpoint(self, trainer):
        checkpoint_path = os.path.join(trainer.args.output_dir, f'{self.model_name}_best_checkpoint')
        print(f'Saving best checkpoint to: {checkpoint_path}')
        trainer.save_model(checkpoint_path)

    def train(self, trainer):
        self.init_wandb_run(f'{self.model_name}_train')

        trainer.args.report_to = "wandb"
        trainer.args.logging_strategy = "epoch"
        trainer.add_callback(WandbCallback())
        trainer.train()
        self.save_best_checkpoint(trainer)

    def evaluate(self, trainer, test_dataset, is_zero_shot=False, init_wandb_run=False):
        if init_wandb_run:
            self.init_wandb_run(f'{self.model_name}_test_only')

        trainer.model.eval()
        p = trainer.predict(test_dataset)
        custom_metrics = self.compute_metrics(p, test_dataset, trainer.tokenizer)
        preds = self.decode_preds(p, trainer.tokenizer)

        model_name = self.model_name.replace('/', '_')
        output_file_name = f'test_results_{model_name}.txt'
        if is_zero_shot:
            output_file_name = f'{model_name}_zero.txt'

        output_path = os.path.join(trainer.args.output_dir, output_file_name)
        output = []
        with open(output_path, 'w') as f:
            f.write('PREDICTED & TARGET\n\n')
            for i in range(len(preds)):
                src = test_dataset[i]['source']
                target = test_dataset[i]['target']
                f.write(f'src: {src}\npred: {preds[i]}\ntarget: {target}\n')
                output.append({'src': src, 'pred': preds[i], 'target': target})
                f.write('-' * 100 + '\n')
            f.write('\n\n\nMETRICS\n\n')
            f.write(f'metrics: {p.metrics}\n')
            f.write(f'custom metrics: {custom_metrics}\n')

        output_df = pd.DataFrame(output)
        output_csv_path = os.path.join(trainer.args.output_dir, f'test_results_{model_name}.csv')
        output_df.to_csv(output_csv_path, index=False)

        wandb.save(output_path)
        wandb.save(output_csv_path)

        print(f'Output (metrics & predictions) saved to: {output_path}')
        print(f'Predictions in csv form are saved to: {output_csv_path}')

    def get_optuna_space(self):
        def optuna_hp_space(trial):
            hpo_params = self.pipeline_config_args["hpo"]["parameters"]
            dict_params = {}
            for param, settings in hpo_params.items():
                if settings["type"] == "float":
                    dict_params[param] = trial.suggest_float(param, settings["min"], settings["max"], log=True)
                elif settings["type"] == "int":
                    dict_params[param] = trial.suggest_int(param, settings["min"], settings["max"], log=True)
                elif settings["type"] == "categorical":
                    dict_params[param] = trial.suggest_categorical(param, settings["values"])
            return dict_params

        return optuna_hp_space

    def hpo(self, trainer):
        res = trainer.hyperparameter_search(
            direction="minimize",
            backend="optuna",
            hp_space=self.get_optuna_space(),
            n_trials=self.pipeline_config_args["hpo"]["nr_trials"],
        )

        best_run_params = res.hyperparameters
        print(f'best run params: {best_run_params}')

        if 'learning_rate' in best_run_params:
            trainer.args.learning_rate = best_run_params['learning_rate']
            print(f'Updated learning rate to: {trainer.args.learning_rate}')
        if 'weight_decay' in best_run_params:
            trainer.args.weight_decay = best_run_params['weight_decay']
            print(f'Updated weight decay to: {trainer.args.weight_decay}')
        if 'num_train_epochs' in best_run_params:
            ###
            # Optuna selects a model based on the last epoch, so varying this parameter is mainly used to avoid
            # choosing the model that is "less prone to overfitting".
            # The trainer however, takes the best model based on the validation loss, so we can train for longer.
            ###
            trainer.args.num_train_epochs = best_run_params['num_train_epochs'] * 2
            print(f'Updated num train epochs to: {trainer.args.num_train_epochs}')
        if 'per_device_train_batch_size' in best_run_params:
            trainer.args.per_device_train_batch_size = best_run_params['per_device_train_batch_size']
            print(f'Updated per device train batch size to: {trainer.args.per_device_train_batch_size}')

        wandb.finish()
        self.init_wandb_run(f'{self.model_name}_hpo_best_run')
        trainer.args.report_to = "wandb"
        trainer.args.logging_strategy = "epoch"
        trainer.add_callback(WandbCallback())

        trainer.train()
        self.save_best_checkpoint(trainer)

        return trainer


class BartBasedModel(RephrasingModel):
    def __init__(self, name: str, device: str, data_path: str, pipeline_config_args: Dict, output_dir: str,
                 max_input_length: int, load_from_checkpoint):
        super().__init__(name, device, data_path, pipeline_config_args, output_dir,
                         max_input_length, load_from_checkpoint)

        self.trainer = self.create_trainer()

    def create_trainer(self):
        def model_init():
            if self.load_from_checkpoint:
                return BartForConditionalGeneration.from_pretrained(self.pipeline_config_args['initial_checkpoint']).to(
                    self.device)
            return BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

        if self.load_from_checkpoint:
            tokenizer = AutoTokenizer.from_pretrained(self.pipeline_config_args['initial_checkpoint'])
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        data = create_datasets(self.data_path, self.pipeline_config_args['max_dups'],
                               self.pipeline_config_args["eval_size"])
        train_set = data['train'].map(self.get_data_preprocessing_func(tokenizer), batched=True)
        eval_set = data['validate'].map(self.get_data_preprocessing_func(tokenizer), batched=True)
        eval_set = eval_set.remove_columns('labels')

        self.test_dataset = data['test'].map(self.get_data_preprocessing_func(tokenizer), batched=True)
        self.test_dataset = self.test_dataset.remove_columns('labels')

        training_args = TrainingArguments(
            output_dir=self.output_dir,
        )

        # Train model
        trainer = Trainer(
            model=None,
            model_init=model_init,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=eval_set,
            tokenizer=tokenizer,
        )

        if self.pipeline_config_args["run_zero_shot"]:
            self.evaluate_bart(is_zero_shot=True)

        return trainer

    def hpo_bart(self):
        updated_trainer = super().hpo(self.trainer)
        self.trainer = updated_trainer

    def train_bart(self):
        super().train(self.trainer)

    def evaluate_bart(self, is_zero_shot=False, init_wandb=False):
        super().evaluate(self.trainer, self.test_dataset, is_zero_shot, init_wandb)


class T5Model(RephrasingModel):
    def __init__(self, name: str, device: str, data_path: str, pipeline_config_args: Dict, output_dir: str,
                 max_input_length: int, load_from_checkpoint):
        super().__init__(name, device, data_path, pipeline_config_args, output_dir,
                         max_input_length, load_from_checkpoint)

        self.trainer = self.create_trainer()

    def decode_preds(self, p: EvalPrediction, tokenizer):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        return preds

    def create_trainer(self):
        if self.load_from_checkpoint:
            tokenizer = AutoTokenizer.from_pretrained(self.pipeline_config_args['initial_checkpoint'])
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def model_init(trial):
            if self.load_from_checkpoint:
                return AutoModelForSeq2SeqLM.from_pretrained(self.pipeline_config_args['initial_checkpoint']).to(
                    self.device)
            return AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        data = create_datasets(self.data_path, self.pipeline_config_args['max_dups'],
                               self.pipeline_config_args["eval_size"])
        train_set = data['train'].map(self.get_data_preprocessing_func(tokenizer), batched=True)
        eval_set = data['validate'].map(self.get_data_preprocessing_func(tokenizer), batched=True)

        self.test_dataset = data['test'].map(self.get_data_preprocessing_func(tokenizer), batched=True)

        args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            bf16=True,
            predict_with_generate=True,
            load_best_model_at_end=True,
            save_total_limit=1,
            save_strategy='epoch',
            evaluation_strategy='epoch',
        )

        trainer = Seq2SeqTrainer(
            model=None,
            model_init=model_init,
            args=args,
            train_dataset=train_set,
            eval_dataset=eval_set,
            tokenizer=tokenizer,
        )

        if self.pipeline_config_args["run_zero_shot"]:
            self.evaluate_t5(is_zero_shot=True)

        return trainer

    def hpo_t5(self):
        updated_trainer = super().hpo(self.trainer)
        self.trainer = updated_trainer

    def train_t5(self):
        super().train(self.trainer)

    def evaluate_t5(self, is_zero_shot=False, init_wandb=False):
        super().evaluate(self.trainer, self.test_dataset, is_zero_shot, init_wandb)


def run_job_bart(args, output_dir):
    model_to_hf_model_name = {
        "bart-detox": "s-nlp/bart-base-detox",
        "bart-large": "facebook/bart-large",
    }

    hf_model_name = model_to_hf_model_name[args.model]
    load_from_checkpoint = args.job_mode == "eval-checkpoint"
    model = BartBasedModel(hf_model_name, args.device, args.data_path, args.rephrasing_pipeline_args,
                           output_dir=output_dir, max_input_length=128, load_from_checkpoint=load_from_checkpoint)
    if args.job_mode == "hpo-and-eval":
        model.hpo_bart()
    if args.job_mode == "train-and-eval":
        model.train_bart()

    init_wandb_on_eval = args.job_mode == "eval-checkpoint"
    model.evaluate_bart(init_wandb=init_wandb_on_eval)


def run_job_t5(args, output_dir):
    model_to_hf_model_name = {
        "t5-formal": "Isotonic/informal_to_formal",
        "t5-detox": "s-nlp/t5-paranmt-detox",
        "t5-large": "t5-large",
        "flan-large": "google/flan-t5-large",
    }

    hf_model_name = model_to_hf_model_name[args.model]
    load_from_checkpoint = args.job_mode == "eval-checkpoint"
    model = T5Model(hf_model_name, args.device, args.data_path, args.rephrasing_pipeline_args,
                    output_dir=output_dir, max_input_length=128, load_from_checkpoint=load_from_checkpoint)

    if args.job_mode == "hpo-and-eval":
        model.hpo_t5()
    if args.job_mode == "train-and-eval":
        model.train_t5()

    init_wandb_on_eval = args.job_mode == "eval-checkpoint"
    model.evaluate_t5(init_wandb=init_wandb_on_eval)


def main():
    parser = argparse.ArgumentParser(description='BART Detox Training')
    parser.add_argument('--config-file', type=str, help='a config json file', default='config.json')

    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)
        for key, value in config.items():
            if isinstance(value, dict) and "value" in value and "choices" in value:
                if value["value"] in value["choices"]:
                    setattr(args, key, value["value"])
                else:
                    print(f"Error: Invalid value '{value['value']}' for '{key}'. "
                          f"Valid choices are: {', '.join(value['choices'])}")
                    return
            else:
                setattr(args, key, value)

    now = str(datetime.now()).replace(' ', '_').replace(':', '_').split('.')[0]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_dir = os.path.join(args.output_dir, now)
    os.makedirs(output_dir, exist_ok=True)

    if args.model.startswith("bart"):
        run_job_bart(args, output_dir)
    elif args.model.startswith("t5") or args.model.startswith("flan"):
        run_job_t5(args, output_dir)


if __name__ == '__main__':
    main()
