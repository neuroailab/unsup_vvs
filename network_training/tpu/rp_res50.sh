python train_combinet.py \
    --expId rp_resnet50 --dataconfig dataset_config_rp.cfg \
    --batchsize 64 --valinum 780 --whichopt 0 --init_lr 0.01 \
    --valdconfig dataset_config_rp.cfg --valid_first 0 \
    --pathconfig rp_resnet50.cfg --nport 27001 \
    --fre_filter 20018 --fre_valid 20018 --ignorebname_new 0 \
    --init_type variance_scaling_initializer \
    --cacheDirPrefix gs://cx_visualmaster/ --namefunc tpu_combine_tfutils_rp --validation_skip 0 \
    --weight_decay 1e-4 --with_feat 0 --tpu_flag 1 --tpu_task rp --withclip 0 "$@"
