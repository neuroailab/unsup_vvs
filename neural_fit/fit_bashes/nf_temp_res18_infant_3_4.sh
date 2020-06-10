for which_split in split_3 split_4
do
    python fit_neural_data.py \
        --whichopt 1 --weight_decay 1e-3 \
        --expId res18_infant_wf_${which_split} --gpu 3 \
        --loadexpId inst_infant_18_fx --cacheDirPrefix /mnt/fs1/chengxuz \
        --cacheDirName neural_fitting_inst \
        --it_nodes encode_[1:1:10],category_2 \
        --v4_nodes encode_[1:1:10],category_2 --batchsize 64 --loadport 27009 \
        --tpu_flag 1 --fre_valid 160 --fre_metric 160 \
        --rp_sub_mean 1 --div_std 1 \
        --network_func get_resnet_18 \
        --network_func_kwargs '{"num_cat": 246}' \
        --ckpt_file /mnt/fs4/chengxuz/tpu_ckpts/res18_infant/model.ckpt-687168 \
        --which_split ${which_split} --train_num_steps 711000
done
