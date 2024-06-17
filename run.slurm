#!/bin/bash
#SBATCH --job-name=pytorch_job
#SBATCH --output=pytorch_job_%j.out
#SBATCH --error=pytorch_job_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --account=your_account

module load python/3.7.4
module load pytorch/1.2.0

# Activate your Python environment, if you have one
# source activate myenv

# Install necessary libraries
pip install --user -r requirements.txt

srun python run.py