"""
This file contains the code for the style classifier.
The code is structured as a single pipeline (for training, evaluating and testing the style classifier model),
which can be run by running this file.
All the parameters have default values (relative to the path of this file),
but can be changed by passing them as arguments to the script (see "help" for more details).
"""

import os
import torch
import numpy as np
import argparse
from datasets import load_dataset, load_metric
from transformers import AutoConfig, AutoTokenizer, Trainer, EvalPrediction, \
    TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding

os.environ['TRANSFORMERS_CACHE'] ='.transformers_cache'
os.environ['HF_HOME'] = '.hf_home'
os.environ['HF_DATASETS_CACHE']= '.hf_datasets_cache'
ACCURACY, MODEL_NAME, SEED, MODEL = 0, 1, 2, 3

# ---------------------------- Fine-tuning pipeline ---------------------------- #

def train_model_pipeline(train_data_path, eval_data_path, test_data_path, results_path, pred_path,  model_name, n_seeds):
    """
    This function fine-tunes the pretrained model on the Corpify dataset.
    :param train_data_path: The path to the training dataset.
    :param eval_data_path: The path to the evaluation dataset.
    :param test_data_path: The path to the test dataset.
    :param results_path: The path to the file to write the results to.
    :param pred_path: The path to the file to write the predictions to.
    :param model_name: The name of the pretrained model to use.
    :param n_seeds: The number of seeds to use for the experiment.
    :return: None
    """
    open(results_path, 'w').close()
    train_dataset = load_dataset('csv', data_files=train_data_path, split='train')
    eval_dataset = load_dataset('csv', data_files=eval_data_path, split='train')
    test_dataset = load_dataset('csv', data_files=test_data_path, split='train')

    # run experiment fon n_seeds
    chosen = [0]
    training_time = 0

    for seed in range(n_seeds):
        trainer, runtime = finetune_model(model_name, seed, train_dataset,
                                          eval_dataset)
        trainer.model.eval()
        evaluation = trainer.evaluate()
        chosen = chosen if chosen[ACCURACY] >= evaluation['eval_accuracy'] else \
            [evaluation['eval_accuracy'], model_name, seed, trainer]
        training_time += runtime
        print(f'Finished model {model_name} with seed {seed}')

        # write model results to file
        with open(results_path, 'a') as f:
            f.write(f'{model_name} with seed {seed} results:\n')
            f.write(f'accuracy - {evaluation["eval_accuracy"]}\n')
            f.write(f'train time - {training_time} seconds\n')

    # save chosen model
    print(f'Chosen model: {chosen[MODEL_NAME]} with seed {chosen[SEED]}')
    chosen[MODEL].save_model(f'model/{chosen[MODEL_NAME]}_{chosen[SEED]}.pt', push_to_hub=False)
    predict_on_test_set(chosen[MODEL_NAME], chosen[SEED], chosen[MODEL], test_dataset, results_path, pred_path)


def finetune_model(model_name, seed, train_dataset, eval_dataset):
    """
    This function fine-tunes the pretrained model on the Corpify dataset.
    :param model_name: The name of the pretrained model to use.
    :param seed: The seed to use for the experiment.
    :param train_dataset: The training dataset.
    :param eval_dataset: The evaluation dataset.
    :return: The fine-tuned model.
    """
    # set training arguments
    training_args = TrainingArguments(seed=seed, disable_tqdm=True,
                                    save_strategy='no',
                                      num_train_epochs=5,
                                      output_dir=f"results/{model_name}_{seed}")
    # tokenize data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = tokenize_data(train_dataset, tokenizer)
    eval_dataset = tokenize_data(eval_dataset, tokenizer)

    # load model
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    data_collator = DataCollatorWithPadding(
        tokenizer=AutoTokenizer.from_pretrained(
            model_name))  # pads to longest sequence in batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    training_run = trainer.train()
    return trainer, training_run.metrics['train_runtime']


def predict_on_test_set(model_name, seed, trainer, test_dataset, results_path, predictions_path):
    """
    This function predicts the labels of the test dataset using the given model.
    :param model_name: The name of the pretrained model to use.
    :param seed: The seed to use for the experiment.
    :param trainer: The fine-tuned model.
    :param test_dataset: The test dataset.
    :param results_path: The path to the file to write the results to.
    :return: None
    """

    test_args = TrainingArguments(seed=seed, per_device_eval_batch_size=1,
                                  disable_tqdm=True,
                                  output_dir=f"results/test_{model_name}_{seed}",
                                  overwrite_output_dir=True)
    trainer.args = test_args
    trainer.model.eval()
    test_dataset = tokenize_data(test_dataset, AutoTokenizer.from_pretrained(model_name))
    test_preds_trainer = trainer.predict(test_dataset)
    test_preds = test_preds_trainer.predictions
    test_preds = np.argmax(test_preds, axis=1)

    # write test results to files
    with open(results_path, 'a') as f:
        f.write(f'predict time,{test_preds_trainer.metrics["test_runtime"]}')
    with open(predictions_path, 'w') as f:
        for input_sentence, label in zip(test_dataset['text'], test_preds):
            f.write(f'{input_sentence} - {"corpy" if label == 1 else "regular"}\n')


# ---------------------------- Helper functions ---------------------------- #

def tokenize_data(dataset, tokenizer):
    """
    Tokenizes the dataset using the given tokenizer, using truncation and no padding.
    :param dataset: dataset to tokenize
    :param tokenizer: tokenizer to use
    :return: the tokenized dataset
    """

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=False)

    return dataset.map(preprocess_function, batched=True)


def compute_metrics(eval_pred):
    """
    Computes the accuracy of the model on the given evaluation prediction.
    :param eval_pred: prediction on the evaluation set
    :return: a dictionary containing the accuracy
    """
    load_accuracy = load_metric('accuracy')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": load_accuracy.compute(predictions=predictions, references=labels)['accuracy']}


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Trains a style classifier on the corpify dataset')
    args.add_argument('--train_data_path', type=str, default='data/train_data.csv')
    args.add_argument('--eval_data_path', type=str, default='data/eval_data.csv')
    args.add_argument('--test_data_path', type=str, default='data/test_data.csv')
    args.add_argument('--results_path', type=str, default='results/classifier_results.txt')
    args.add_argument('--pred_path', type=str, default='results/classifier_predictions.txt')
    args.add_argument('--nseeds', type=int, default=3)
    args.add_argument('--model', type=str, default="roberta-base")
    args = args.parse_args()
    train_model_pipeline(args.train_data_path, args.eval_data_path, args.test_data_path, args.results_path, args.pred_path, args.model, args.nseeds)
