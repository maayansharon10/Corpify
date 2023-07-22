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

### Job Modes

TODO




