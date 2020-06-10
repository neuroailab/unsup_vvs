for which_split in 0 1 2 3 4
do
    python main_pt_official.py \
        --data /mnt/fs4/chengxuz/v4it_temp_results/var6_tfrecords/split_${which_split}/images \
        --save_path /mnt/fs4/chengxuz/v4it_temp_results/pt_official_res18_nf/hvm_var6_split_${which_split} \
        --dataset_type hvm_var6
done
