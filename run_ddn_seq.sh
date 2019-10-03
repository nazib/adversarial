#!/bin/bash -l
#PBS -N 25%_DDN_removing_BN
#PBS -l walltime=200:00:00
#PBS -l mem=64Gb
#PBS -l ncpus=1
#PBS -l ngpus=1
#PBS -l gputype=M40

#module load cudnn/7-cuda-9.0.176
#module load python/2.7.13-foss-2017a-foss
#python/2.7.13-foss-2017a-foss
cd patchvm
module load tensorflow/1.12.0-gpu-m40-foss-2016a-python-2.7.12
python new_train.py 25%_configh5.ini




