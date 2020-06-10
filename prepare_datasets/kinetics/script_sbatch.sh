#len_idx=250
len_tfr=1
#len_idx=200
#len_tfr=5

#for sta_idx in $(seq 0 ${len_idx} 246535)
#for sta_idx in $(seq 0 ${len_idx} 19907)
#for sta_idx in $(seq 0 ${len_tfr} 550)
for sta_idx in $(seq 0 ${len_tfr} 46)
#for sta_idx in 0
do
    sbatch script_tfrs.sh ${sta_idx} ${len_tfr}
    #sbatch script_download.sh ${sta_idx}
    #sbatch script_dwnsmpl.sh ${sta_idx} ${len_idx}
done
