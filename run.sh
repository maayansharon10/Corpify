#!/bin/bash -x
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a10:1
#SBATCH --mem-per-cpu=30g

echo $PWD
activate () {
    . $PWD/myenv/bin/activate
}

set_env_vars () {
  TRANSFORMERS_CACHE=$PWD/.trans_cache
  export TRANSFORMERS_CACHE

  HF_DATASETS_CACHE=$PWD/.datasets_cache
  export HF_DATASETS_CACHE

  HF_HOME=$PWD/.hf_home
  export HF_HOME
}

activate
set_env_vars

python3 main.py --data-path 'Data/all_sent_raw.csv' --device 'cuda'