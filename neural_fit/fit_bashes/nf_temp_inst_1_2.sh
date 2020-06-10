for which_split in split_0 split_1 split_2
do
    python fit_neural_data.py \
        --whichopt 1 --weight_decay 1e-3 --expId inst_res18_wf_${which_split} \
        --gpu 0 --loadexpId inst_res18_newp_dset \
        --cacheDirPrefix /mnt/fs1/chengxuz \
        --cacheDirName neural_fitting_inst  --pathconfig instance_resnet18.cfg \
        --it_nodes encode_[1:1:10],category_2 \
        --v4_nodes encode_[1:1:10],category_2 \
        --batchsize 64 --loadport 27009 --tpu_flag 1 \
        --fre_valid 160 --fre_metric 160 --rp_sub_mean 1 --div_std 1 \
        --which_split ${which_split} --train_num_steps 2585000
done
