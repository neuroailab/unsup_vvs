fre_filter=20020
#fre_filter=100
test_str="
python train_combinet.py --expId tpu_mt101_p01_rdlc_rpcol_rl \
    --dataconfig dataset_config_image_wun.cfg --batchsize 128 --valinum 400 --init_lr 0.1 \
    --pathconfig mean_teacher_resnet101_block3.cfg --mean_teacher 1 --whichopt 3 --nport 27009 \
    --fre_filter ${fre_filter} --fre_valid ${fre_filter} --init_type variance_scaling_initializer \
    --val_n_threads 4 --cacheDirPrefix gs://cx_visualmaster/ --namefunc tpu_combine_tfutils_general \
    --tpu_task mean_teacher --weight_decay 5e-5 --tpu_name cx-cbnet-5 --with_feat 0 --tpu_flag 1 \
    --cons_ramp_len 50000 --whichimagenet 10 --withclip 0 \
    --mt_ramp_down 1 --color_dp_tl 1 --tpu_center_im 0 --combine_col_rp 1  \
    --ema_zerodb 1 --add_batchname _rp --ignorebname_new 0 \
"

for step in $(seq 1 30)
#for step in 1
do
    ${test_str} --valid_first 0
    ${test_str} --valid_first 1
done
