#!/bin/bash -l
#PBS -N 100%_GAN
#PBS -l walltime=100:00:00
#PBS -l mem=100Gb
#PBS -l ncpus=1
#PBS -l ngpus=1
#PBS -l gputype=M40

module load cudnn/7-cuda-9.0.176
module load python/3.6.4-foss-2016a
cd adversarial
python train_100.py 100%_config.ini

