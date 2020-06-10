test_str="
python train_combinet.py --expId tpu_resnet18_mtp05_inst_infant \
    --dataconfig dataset_config_image_wun.cfg \
    --batchsize 256 --valinum 196 --init_lr 0.1 \
    --network_func get_mean_teacher_resnet_18 \
    --mean_teacher 1 --whichopt 3 --nport 27001 \
    --fre_filter 10010 --fre_valid 10010 \
    --init_type variance_scaling_initializer \
    --cacheDirPrefix gs://cx_visualmaster/ \
    --namefunc tpu_combine_tfutils_general \
    --tpu_task mean_teacher --weight_decay 5e-5 \
    --with_feat 0 --tpu_flag 1 \
    --cons_ramp_len 25000 \
    --color_norm 1 \
    --whichimagenet infant_05 --withclip 0 \
    --mt_ramp_down 1 --ema_zerodb 1 \
    --mt_infant_loss 1 \
"

for step in $(seq 1 30)
do
    ${test_str} --valid_first 0 --network_func_kwargs '{"num_cat": 246}' "$@"
    ${test_str} --valid_first 1 --network_func_kwargs '{"num_cat": 246}' "$@"
done
