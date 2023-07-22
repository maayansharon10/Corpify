#!/bin/bash -x
#SBATCH --time=3:00:00
#SBATCH -c10
#SBATCH --mem=50g

PWD=$(pwd)
activate() {
  . $PWD/myenv/bin/activate
}

# Create a virtual environment:
python3 -m venv myenv
activate

# Install packages:
curl -sS https://bootstrap.pypa.io/get-pip.py | python3
pip install -r requirements.txt