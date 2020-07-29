test_str="
python train_combinet.py --expId tpu_resnet101_mt_rd_lc_infant_ctl \
    --dataconfig dataset_config_image_wun.cfg --batchsize 128 --valinum 400 --init_lr 0.1 \
    --pathconfig mean_teacher_resnet101_block3.cfg --mean_teacher 1 --whichopt 3 --nport 27009 \
    --fre_filter 20020 --fre_valid 20020 --init_type variance_scaling_initializer \
    --val_n_threads 4 --cacheDirPrefix gs://siming-model-5/ --namefunc tpu_combine_tfutils_general \
    --tpu_task mean_teacher --weight_decay 5e-5 --tpu_name m-tpu-7 --with_feat 0 --tpu_flag 1 \
    --cons_ramp_len 50000 --whichimagenet infant_ctl --withclip 0 \
    --mt_ramp_down 1 --color_dp_tl 1 --combine_col_rp 1 --mt_infant_loss 0  \
"

for step in $(seq 1 30)
do
    ${test_str} --valid_first 0
    ${test_str} --valid_first 1
done
