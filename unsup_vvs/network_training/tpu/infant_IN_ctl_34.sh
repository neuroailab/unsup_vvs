test_str="
python train_combinet.py --expId tpu_new_IN_ctl_res34 \
    --dataconfig dataset_config_image.cfg --batchsize 256 \
    --valinum 196 --nport 27001 \
    --pathconfig newIN_resnet34.cfg \
    --whichopt 3 \
    --fre_filter 9544 --fre_valid 9544 \
    --init_type variance_scaling_initializer \
    --val_n_threads 4 --cacheDirPrefix gs://cx_visualmaster/ \
    --namefunc tpu_combine_tfutils_general \
    --tpu_task imagenet --weight_decay 1e-4 \
    --tpu_name cx-cbnet-14 --with_feat 0 --tpu_flag 1 \
    --whichimagenet infant_ctl --withclip 0 --color_norm 1 \
"

for step in $(seq 0 30)
do
    ${test_str} --valid_first 0 --init_lr 0.1
    ${test_str} --valid_first 1
done

for step in $(seq 0 30)
do
    ${test_str} --valid_first 0 --init_lr 0.01
    ${test_str} --valid_first 1
done

for step in $(seq 0 30)
do
    ${test_str} --valid_first 0 --init_lr 0.001
    ${test_str} --valid_first 1
done
