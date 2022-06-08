#!/bin/bash

END=130
for i in $(seq 0 $END);
do
    sbatch --partition=gpu2019 --mem=233000M bootstrapping_call.sh $i sites
done

END=199
for i in $(seq 130 $END);
do
    sbatch --partition=cpu2019 --mem=250000M bootstrapping_call.sh $i sites
done

END=130
for i in $(seq 0 $END);
do
    sbatch --partition=gpu2019 --mem=233000M bootstrapping_call.sh $i years
done

END=199
for i in $(seq 130 $END);
do
    sbatch --partition=cpu2019 --mem=250000M bootstrapping_call.sh $i years
done


