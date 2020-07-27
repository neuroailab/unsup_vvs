python train_combinet.py \
    --expId combine_rp_col_res18_fx --dataconfig dataset_config_rpcol.cfg \
    --batchsize 32 --valinum 200 --whichopt 1 --init_lr 0.01 \
    --valdconfig dataset_config_rpcol.cfg --valid_first 0 \
    --pathconfig rpcol_resnet18.cfg --nport 27001 --fre_filter 20018 --fre_valid 20018 \
    --ignorebname_new 0 --init_type variance_scaling_initializer --val_n_threads 4 \
    --cacheDirPrefix gs://cx_visualmaster/ --sm_loaddir gs://infant_imagenet/new_imagenet_tfr/ \
    --namefunc tpu_combine_tfutils_rp_col --weight_decay 1e-4 --with_feat 0 \
    --tpu_flag 1 --tpu_task combine_rp_col --add_batchname _rp --use_lasso 0 --num_grids 1 "$@"
