#!/bin/bash
#SBATCH --job-name=pytorch_job
#SBATCH --output=pytorch_job_%j.out
#SBATCH --error=pytorch_job_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

#module load python/3.8.0
pip install --user scikit-learn

python train_model.py