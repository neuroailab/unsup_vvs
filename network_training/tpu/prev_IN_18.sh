#python train_combinet.py --expId tpu_prev_IN_res34_rpbwd_gwd \
test_str="
python train_combinet.py --expId tpu_prev_IN_res18_fx \
    --dataconfig dataset_config_image.cfg --batchsize 256 \
    --valinum 196 --nport 27001 \
    --network_func get_resnet_18 \
    --whichopt 0 \
    --fre_filter 10009 --fre_valid 10009 \
    --init_type variance_scaling_initializer \
    --cacheDirPrefix gs://cx_visualmaster/ \
    --namefunc tpu_combine_tfutils_general \
    --tpu_task imagenet --global_weight_decay 1e-4 \
    --tpu_name cx-cbnet-0 --with_feat 0 --tpu_flag 1 \
    --whichimagenet full --withclip 0 --color_norm 1 --resnet_prep \
"

for step in $(seq 0 15)
do
    ${test_str} --valid_first 0 --init_lr 0.1
    ${test_str} --valid_first 1
done

for step in $(seq 0 15)
do
    ${test_str} --valid_first 0 --init_lr 0.01
    ${test_str} --valid_first 1
done

for step in $(seq 0 15)
do
    ${test_str} --valid_first 0 --init_lr 0.001
    ${test_str} --valid_first 1
done

for step in $(seq 0 15)
do
    ${test_str} --valid_first 0 --init_lr 0.0001
    ${test_str} --valid_first 1
done
