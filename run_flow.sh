#!/bin/bash -l
#PBS -N 100%_patch_extract
#PBS -l walltime=100:00:00
#PBS -l mem=250Gb
#PBS -l ncpus=1


module load cudnn/7-cuda-9.0.176
module load python/3.6.4-foss-2016a
cd adversarial
python ExtractPatchesHigh.py

