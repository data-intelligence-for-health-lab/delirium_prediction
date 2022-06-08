#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --partition=cpu2019
#SBATCH --mem=500000M
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --output=/project/M-ABeICU176709/delirium/code/revision/log/shap.out
#SBATCH --error=/project/M-ABeICU176709/delirium/code/revision/log/shap.err

srun --ntasks=1 --mem=500000M python 14_shap_values.py

