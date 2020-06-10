python train_combinet.py \
    --expId col_res50 --dataconfig dataset_config_col.cfg \
    --batchsize 64 --valinum 780 --whichopt 1 \
    --init_lr 0.00001 --valdconfig dataset_config_col.cfg \
    --valid_first 0 --pathconfig col_resnet50.cfg \
    --nport 27001 --withclip 0 --fre_filter 20018 --fre_valid 20018 \
    --ignorebname_new 0 --init_type variance_scaling_initializer \
    --cacheDirPrefix gs://cx_visualmaster/ \
    --namefunc tpu_combine_tfutils_col --weight_decay 1e-4 --with_feat 0 \
    --tpu_flag 1 --tpu_task colorization --adameps 1e-8 --col_down 8 --col_knn 1 "$@"
