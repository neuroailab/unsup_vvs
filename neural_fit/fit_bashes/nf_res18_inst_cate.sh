python fit_neural_data.py \
    --whichopt 1 --weight_decay 1e-3 \
    --expId inst_cate_res18_2 --gpu 2 \
    --cacheDirName neural_fitting_inst \
    --it_nodes encode_[1:1:10] \
    --v4_nodes encode_[1:1:10] \
    --batchsize 64 --loadport 27009 \
    --tpu_flag 1 --fre_valid 160 --fre_metric 160 \
    --rp_sub_mean 1 --div_std 1 \
    --network_func get_resnet_18_inst_and_cate \
    --use_dataset_inter \
    --loadexpId res18_inst_cate_1 \
    --loaddbname combine_instance --train_num_steps 1246385
    #--expId inst_cate_res18 --gpu 2 \
