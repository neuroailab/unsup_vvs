gpu=$1
split=$2
for rep in 0 1 2
do
    python fit_neural_data.py --gpu ${gpu} \
        --load_setting_func v1v2_fx_setting.cate_res18_v1v2_bwd_fx_to_be_filled \
        --expId fx_cate_e49_bwd_sp${split}_${rep} \
        --which_split split_${split}

    python fit_neural_data.py --gpu ${gpu} \
        --load_setting_func v1v2_fx_setting.cate_res18_v1v2_bbwd_fx_to_be_filled \
        --expId fx_cate_e49_bbwd_sp${split}_${rep} \
        --which_split split_${split}

    for layer in 1 2 3
    do
        python fit_neural_data.py --gpu ${gpu} \
            --load_setting_func v1v2_fx_setting.cate_res18_v1v2_fx_e${layer}_to_be_filled \
            --expId fx_cate_e${layer}_sp${split}_${rep} \
            --which_split split_${split}
    done
done
