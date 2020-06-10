#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -c 1
#SBATCH --output=/mnt/fs1/chengxuz/slurm_out_all/kinetics_dwnsmpl_%j.out
#SBATCH --exclude=node11-neuroaicluster,node5-neuroaicluster,node12-neuroaicluster

#python dwnsmpl_videos.py --sta_avi ${1} --len_avi ${2} --check 1
python dwnsmpl_videos.py --sta_avi ${1} --len_avi ${2} --check 0 --avidir /mnt/fs1/Dataset/kinetics/vd_dwnld_val --dwnsmpldir /mnt/fs1/Dataset/kinetics/vd_dwnld_val_5fps
