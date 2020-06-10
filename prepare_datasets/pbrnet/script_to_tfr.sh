#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -p owners
#SBATCH --output=/scratch/users/chengxuz/slurm_output/pbrnet_tfrecs_%j.out

module load tensorflow/0.12.1
module load anaconda/anaconda.4.2.0.python2.7

python write_tfrs.py --includelen 100 --indxlen 5 "$@"
