gpu=$1
for split in 0 1 2 3 4
do
    python fit_neural_data.py --gpu ${gpu} \
        --load_setting_func jrnl_setting.ir_vm_res18_v1_tc_unfilled \
        --expId ir_vm_res18_v1_sp${split} \
        --which_split split_${split}

    python fit_neural_data.py --gpu ${gpu} \
        --load_setting_func jrnl_setting.ir_vm_res18_v1_tc_swd_unfilled \
        --expId ir_vm_res18_v1_swd_sp${split} \
        --which_split split_${split}
done
