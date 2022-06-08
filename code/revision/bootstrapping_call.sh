#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --output=/project/M-ABeICU176709/delirium/code/revision/log/job_%j.out
#SBATCH --error=/project/M-ABeICU176709/delirium/code/revision/log/job_%j.err

srun --ntasks=1 python 07_bootstrapping.py $1 $2
