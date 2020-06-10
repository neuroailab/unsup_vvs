#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -c 1
#SBATCH --output=/mnt/fs0/chengxuz/slurm_out_all/imagenet_%j.out

#python combine_tfr.py --staindx ${1} --lenindx ${2}
python split_tfr.py --staindx ${1} --lenindx ${2}
