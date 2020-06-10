#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -c 1
#SBATCH --output=/mnt/fs0/chengxuz/slurm_out_all/kinetics_download_%j.out
#SBATCH --exclude=node11-neuroaicluster,node5-neuroaicluster,node12-neuroaicluster

#python write_tfrs.py --tfr_idx ${1} --len_tfr ${2} --check 1
#python write_tfrs.py --tfr_idx ${1} --len_tfr ${2} --check 0 --write_avi_path 1
#python write_tfrs.py --tfr_idx ${1} --len_tfr ${2} --check 0 --csvpath /mnt/fs1/Dataset/kinetics/kinetics_val.csv --tfrdir /mnt/fs1/Dataset/kinetics/val_tfrs --avidir /mnt/fs1/Dataset/kinetics/vd_dwnld_val
#python write_tfrs.py --tfr_idx ${1} --len_tfr ${2} --check 0 --write_avi_path 1 --csvpath /mnt/fs1/Dataset/kinetics/kinetics_val.csv --tfrdir /mnt/fs1/Dataset/kinetics/val_tfrs --avidir /mnt/fs1/Dataset/kinetics/vd_dwnld_val
#python write_tfrs.py --tfr_idx ${1} --len_tfr ${2} --check 0 --write_avi_path 1 --avidir /mnt/fs1/Dataset/kinetics/vd_dwnld_5fps --tfrdir /mnt/fs1/Dataset/kinetics/train_tfrs_5fps
python write_tfrs.py --tfr_idx ${1} --len_tfr ${2} --check 0 --write_avi_path 1 --csvpath /mnt/fs1/Dataset/kinetics/kinetics_val.csv --tfrdir /mnt/fs1/Dataset/kinetics/val_tfrs_5fps --avidir /mnt/fs1/Dataset/kinetics/vd_dwnld_val_5fps
