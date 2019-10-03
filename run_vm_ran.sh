#!/bin/bash -l
#PBS -N VM_RANDOM
#PBS -l walltime=24:00:00
#PBS -l mem=64Gb
#PBS -l ncpus=1
#PBS -l ngpus=1

module load cudnn/7-cuda-9.0.176
module load python/2.7.11-foss-2016a
#python/2.7.13-foss-2017a-foss
cd patchvm
python new_train.py conf_patchvm_ran


