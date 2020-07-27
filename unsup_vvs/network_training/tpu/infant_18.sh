test_str="
python train_combinet.py --expId tpu_infant_res18 \
    --dataconfig dataset_config_image.cfg --batchsize 256 \
    --valinum 196 --nport 27001 \
    --network_func get_resnet_18 \
    --whichopt 0 \
    --fre_filter 19088 --fre_valid 19088 \
    --init_type variance_scaling_initializer \
    --cacheDirPrefix gs://cx_visualmaster/ \
    --namefunc tpu_combine_tfutils_general \
    --tpu_task imagenet --global_weight_decay 1e-4 \
    --with_feat 0 --tpu_flag 1 \
    --whichimagenet infant --withclip 0 --color_norm 1 --resnet_prep \
"

#for init_lr in 0.1 0.01 0.001 0.0001
for init_lr in 0.01 0.001 0.0001
do
    for step in $(seq 0 8)
    do
        ${test_str} --valid_first 0 --init_lr ${init_lr} --network_func_kwargs '{"num_cat": 246}' "$@"
        ${test_str} --valid_first 1 --network_func_kwargs '{"num_cat": 246}' "$@"
    done
done
