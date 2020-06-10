cd ~/deepcluster

for which_split in 0 1 2 3 4 5 6 7 8 9
do
    python output_hvm.py --data /mnt/fs4/chengxuz/v1_cadena_related/tfrs/split_${which_split}/images \
        --model /mnt/fs4/chengxuz/deepcluster_models/res18_np/checkpoint.pth.tar \
        --conv 104,105,106,107,108,109,110,111,112 --dataset_type v1_tc \
        --save_path /mnt/fs4/chengxuz/v4it_temp_results/deepcluster_hvm/res18_np/V1_split_${which_split} --resnet_deseq
done
