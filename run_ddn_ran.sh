#!/bin/bash -l
#PBS -N 10%_advers_CC+fake
#PBS -l walltime=100:00:00
#PBS -l mem=64Gb
#PBS -l ncpus=1
#PBS -l ngpus=1
#PBS -l gputype=M40

module load cudnn/7-cuda-9.0.176
module load python/3.6.4-foss-2016a
cd adversarial
python train_tf.py 25%_config.ini




