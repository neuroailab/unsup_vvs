for which_split in 0 1 2 3 4
do
    python main_pt_official.py \
        --data /mnt/fs4/chengxuz/v1_cadena_related/tfrs/split_${which_split}/images \
        --save_path /mnt/fs4/chengxuz/v4it_temp_results/pt_official_res18_nf/V1_split_${which_split} \
        --dataset_type v1_tc
done
