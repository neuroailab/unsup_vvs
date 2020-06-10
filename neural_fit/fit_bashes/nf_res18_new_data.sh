python fit_neural_data.py \
    --whichopt 1 --weight_decay 1e-3 \
    --expId inst_res18_IN_wf_new_data_2 --gpu 1 \
    --loadexpId inst_infant_18_fx \
    --cacheDirName neural_fitting_inst \
    --it_nodes encode_[1:1:10],category_2 \
    --v4_nodes encode_[1:1:10],category_2 --batchsize 64 --loadport 27009 \
    --tpu_flag 1 --fre_valid 160 --fre_metric 160 \
    --rp_sub_mean 1 --div_std 1 \
    --network_func get_resnet_18 \
    --use_dataset_inter \
    --ckpt_file /mnt/fs4/chengxuz/tpu_ckpts/res18/model.ckpt-640576 \
    --train_num_steps 665576
    #--expId inst_res18_IN_wf_new_data --gpu 1 \
