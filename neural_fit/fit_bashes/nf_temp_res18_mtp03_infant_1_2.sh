for which_split in split_0 split_1 split_2
do
    python fit_neural_data.py \
        --whichopt 1 --weight_decay 1e-3 \
        --expId res18_mtp03_infant_wf_${which_split} --gpu 3 \
        --loadexpId inst_infant_18_fx --cacheDirPrefix /data5/chengxuz \
        --cacheDirName neural_fitting_inst \
        --it_nodes ema_encode_[1:1:10],ema_category_2 \
        --v4_nodes ema_encode_[1:1:10],ema_category_2 --batchsize 64 --loadport 27009 \
        --tpu_flag 1 --fre_valid 160 --fre_metric 160 \
        --rp_sub_mean 1 --div_std 1 \
        --network_func get_mean_teacher_resnet_18 \
        --network_func_kwargs '{"num_cat": 246}' \
        --mean_teacher 1 \
        --ckpt_file /home/chengxuz/tpu_ckpts/res18_mt_p03_infant/model.ckpt-60060 \
        --which_split ${which_split} --train_num_steps 85000
done
