for tpu_num in $(seq 0 6)
do
    python train_combinet.py --expId combine_rdc_no_lasso_0 --dataconfig dataset_config_rdc.cfg --batchsize 16 --valinum 1 --whichopt 1 --init_lr 0.01 --valdconfig dataset_config_rdc.cfg --valid_first 0 --pathconfig rdc_resnet101_block3_nolasso.cfg --nport 27009 --withclip 1 --fre_filter 10000 --fre_valid 10000 --ignorebname_new 0 --init_type variance_scaling_initializer --val_n_threads 4 --cacheDirPrefix gs://siming-model-5/ --namefunc tpu_combine_tfutils_rdc --weight_decay 1e-4 --with_feat 0 --tpu_flag 1 --tpu_task combine_rdc --sm_loaddir gs://siming-dataset-1/pbrnet/ --sm_loaddir2 gs://siming-dataset-1/ --tpu_name cb-tpu-$tpu_num --add_batchname _rp --use_lasso 0 --num_grids 1 --withclip 0 --validation_skip 1
    if [$tpu_num==5]
    then
        $tpu_num=0
    fi
done
