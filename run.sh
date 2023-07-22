#!/bin/bash -x
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem-per-cpu=10g

echo $PWD
activate() {
  . $PWD/myenv/bin/activate
}

set_env_vars() {
  TRANSFORMERS_CACHE=$PWD/.trans_cache
  export TRANSFORMERS_CACHE

  HF_DATASETS_CACHE=$PWD/.datasets_cache
  export HF_DATASETS_CACHE

  HF_HOME=$PWD/.hf_home
  export HF_HOME
}

activate
set_env_vars
# wandb disabled


python3 main.py --config-file config.json
