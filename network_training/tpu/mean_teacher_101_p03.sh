test_str="
python train_combinet.py --expId tpu_resnet101_mtp03_ub \
    --dataconfig dataset_config_image_wun.cfg --batchsize 128 --valinum 400 --init_lr 0.1 \
    --pathconfig mean_teacher_resnet101_block3.cfg --mean_teacher 1 --whichopt 3 --nport 27009 \
    --fre_filter 20020 --fre_valid 20020 --init_type variance_scaling_initializer \
    --val_n_threads 4 --cacheDirPrefix gs://cx_visualmaster/ --namefunc tpu_combine_tfutils_general \
    --tpu_task mean_teacher --weight_decay 5e-5 --tpu_name cx-cbnet-8 --with_feat 0 --tpu_flag 1 \
    --cons_ramp_len 50000 --whichimagenet 11 --withclip 0 \
"

for step in $(seq 1 30)
do
    ${test_str} --valid_first 0
    ${test_str} --valid_first 1
done
