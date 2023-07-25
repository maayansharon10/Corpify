# Corpify

TODO: add general description

## Setup

1. Clone the repository:

   ``` git clone https://github.com/maayansharon10/Corpify.git ```

2. Enter the repository:

   ```cd Corpify```

3. Create a virtual environment:

   ``` python3 -m venv myenv ```

4. Activate the virtual environment:

   ``` source myenv/bin/activate ```

5. Install requirements:

   ``` pip install -r requirements.txt ```

Alternatively, on cluster, you can run the build_venv.sh script, which will do 3..5 for you:

```chmod +x ./build_env.sh```

```sbatch ./build_env.sh ```

## Running New Jobs

### Using Weights and Biases

We maintain comprehensive documentation for all our experiments using Weights and Biases. To get started with running a
new experiment, please follow these steps:

1. Register and create a new project on Weights and Biases by following the instructions in the quickstart
   guide: https://docs.wandb.ai/quickstart.
2. After creating the project, log in to your Weights and Biases account.
3. In the config.json file, include your project name in the following format:

```   
"training": {
...
"wandb_project": "YOUR-PROJECT-NAME",
...

```

To bypass the use of Weights and Biases entirely, you can run the following command inside the virtual environment:

```wandb disabled```

If you are working on the cluster, remember to uncomment the relevant line in run.sh to disable Weights and Biases.

### Running a Job

All jobs are defined using a configuration file in JSON format, which contains all the necessary parameters for the job.
An example configuration file can be found in `config.json`. To execute the job, the configuration file is passed as an
argument to the `main.py` script:

``` python3 main.py --config-file config.json ```

For cluster environments, an alternative method is available:

1. Make the run script executable:

   ```chmod +x ./run.sh```

2. Submit the job to the cluster using the run script:

   ```sbatch ./run.sh```

The run.sh script takes care of activating the virtual environment, adjusting the default cache path of Hugging-Face
libraries, and running main.py with config.json as the configuration file.

### Available Models

We currently support the following models as a base for fine-tuning:

1. **t5-large** (https://huggingface.co/t5-large): the default T5-large model.
2. **t5-detox** (https://huggingface.co/s-nlp/t5-paranmt-detox): T5 fine-tuned on ParaNMT (a dataset of English-English
   paraphrasing, filtered for the task of detoxification).
3. **t5-formal** (https://huggingface.co/Isotonic/informal_to_formal): T5-base fine-tuned on the GYAFC (informal-formal)
   dataset
4. **flan-large** (https://huggingface.co/google/flan-t5-large): the default FLAN-large model.
5. **bart-large** (https://huggingface.co/facebook/bart-large): the default BART-large model.
6. **bart-detox** (https://huggingface.co/s-nlp/bart-paranmt-detox): BART-base trained on the ParaDetox (
   toxic→not-toxic) dataset.

The model is defined in the configuration file under the `model` key. For example:

```
"model": {
 "value": "t5-large",
 "choices": [
   "t5-detox",
   "t5-formal",
   "t5-large",
   "bart-detox",
   "bart-large",
   "flan-large"
 ]
}
```

### Job Modes

The job mode is defined in the configuration file under the `job_mode` key. The following modes are available:

#### train-and-eval

The model is trained and evaluated, using the default hyperparameters.

#### hpo-and-eval
*Note: This mode is not supported for BART models.*

In this job mode, hyperparameter optimization is performed using the Optuna library. The supported hyperparameters are:

* weight_decay
* num_train_epochs
* per_device_train_batch_size
* learning_rate

The allowed values for each hyperparameter are defined in the configuration file under the `hpo` key. For example:

```
"learning_rate": {
 "type": "float",
 "min": 1e-05,
 "max": 1e-02
},
```

Optuna uses a random grid search by default, meaning that it performs multiple attempts to obtain the best
hyperparameters (trials). It selects the hyperparameters for each trial randomly from the allowed values defined in the
configuration file. The number of trials is defined in the configuration file under the `hpo_trials` key. For example:

```
"hpo_trials": 10,
```

The best trial is selected based on the evaluation loss, and a new training and evaluation session is performed using
the best hyperparameters.

#### eval-checkpoint

This job mode is used to evaluate a model from a given checkpoint, without additional training. checkpoint. The path to
the checkpoint is defined in the configuration file under the `initial_checkpoint` key. For example:

```
"initial_checkpoint": "./results/2023-07-22_13_04_16/t5-large_best_checkpoint",
```

#### eval-zero-shot

The default model checkpoint is downloaded from Hugging-Face and evaluated on the test set.

### Other Controllable Parameters

```
 "max_dups": 1,
 "eval_size": 0.2,
```

`max_dups` is the maximum number of examples with the same source sentence allowed in the data.
`eval_size` is the portion of the data to be used for evaluation. Half of it is used for creating the dev-set and the
rest is used for the test-set.

