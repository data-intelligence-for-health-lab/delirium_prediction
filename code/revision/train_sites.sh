#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu2019
#SBATCH --gres=gpu:1
#SBATCH --mem=700000M
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --output=/project/M-ABeICU176709/delirium/code/revision/log/sites.out
#SBATCH --error=/project/M-ABeICU176709/delirium/code/revision/log/sites.err

srun --ntasks=1 --mem=700000M --gres=gpu:1 python 05_train_sites.py

