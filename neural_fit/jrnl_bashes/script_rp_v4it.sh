gpu=$1
for split in 0 1 2 3 4
do
    python fit_neural_data.py --gpu ${gpu} \
        --load_setting_func jrnl_setting.rp_res18_v4it_unfilled \
        --expId rp_res18_v4it_sp${split} \
        --which_split split_${split}

    python fit_neural_data.py --gpu ${gpu} \
        --load_setting_func jrnl_setting.rp_res18_v4it_swd_unfilled \
        --expId rp_res18_v4it_swd_sp${split} \
        --which_split split_${split}
done
