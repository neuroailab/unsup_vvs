test_str="
python train_combinet.py --expId tpu_resnet152_mtcrsb_2 \
    --dataconfig dataset_config_image_wun.cfg --batchsize 64 --valinum 800 --init_lr 0.1 \
    --pathconfig resnet152_mean_teacher.cfg --mean_teacher 1 --whichopt 3 --nport 27009 \
    --fre_filter 21020 --fre_valid 21020 --init_type variance_scaling_initializer \
    --val_n_threads 4 --cacheDirPrefix gs://cx_visualmaster/ --namefunc tpu_combine_tfutils_general \
    --tpu_task mean_teacher --weight_decay 5e-5 --tpu_name cx-cbnet-5 --with_feat 0 --tpu_flag 0 \
    --cons_ramp_len 100000 --whichimagenet 6 --withclip 0 \
    --no_shuffle 1 \
"

for step in $(seq 1 30)
do
    ${test_str} --valid_first 0
    ${test_str} --valid_first 1
done
