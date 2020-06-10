indxlen=5
for indxnow in $(seq ${indxlen} ${indxlen} 415)
do
    sbatch script_to_tfr.sh --indx ${indxnow}
    sbatch script_to_tfr.sh --keyname category --indx ${indxnow}
    sbatch script_to_tfr.sh --keyname depth --filetype uint16 --indx ${indxnow}
    sbatch script_to_tfr.sh --keyname normal --pngpattern norm --loaddir /scratch/users/chengxuz/Data/pbrnet/normal --savedir /scratch/users/chengxuz/Data/pbrnet/tfrecords/normal --indx ${indxnow}
    sbatch script_to_tfr.sh --keyname valid --pngpattern valid --loaddir /scratch/users/chengxuz/Data/pbrnet/normal --savedir /scratch/users/chengxuz/Data/pbrnet/tfrecords/valid --indx ${indxnow}
done
