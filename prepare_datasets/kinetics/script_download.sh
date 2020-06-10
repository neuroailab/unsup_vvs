#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -c 1
#SBATCH --output=/mnt/fs1/chengxuz/slurm_out_all/kinetics_download_%j.out

#python script_download.py --sta_idx ${1} --check 1
python script_download.py --sta_idx ${1} --check 0 --csvpath /mnt/fs1/Dataset/kinetics/kinetics_val.csv --savedir /mnt/fs1/Dataset/kinetics/vd_dwnld_val --len_idx 250
