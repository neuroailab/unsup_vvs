#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -c 1
#SBATCH -p yamins
#SBATCH --output=/scratch/users/chengxuz/slurm_output/imagenet_%j.out

module load tensorflow/0.12.1
module load anaconda/anaconda.4.2.0.python2.7

python write_tfrs.py --staindx ${1} --lenindx ${2} --checkblack 1
#python write_tfrs.py --staindx ${1} --lenindx ${2} --savedir /scratch/users/chengxuz/Data/imagenet_tfr/tfrs_val --sshfolder /mnt/fs1/Dataset/imagenet_again/tfr_val --txtprefix /scratch/users/chengxuz/Data/imagenet_devkit/val_fname_
