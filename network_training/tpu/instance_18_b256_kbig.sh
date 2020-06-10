test_str="
python train_combinet.py --expId tpu_inst_resnet18_kb \
     --dataconfig dataset_config_image.cfg --valinum 782 --batchsize 256 --valbatchsize 64 --label_norm 1 --nport 27001 \
     --withclip 0 --no_shuffle 0 --init_lr 0.03 --fre_filter 5004 --pathconfig instance_resnet18.cfg \
     --global_weight_decay 1e-4 --with_feat 0 --color_norm 1 --fre_valid 5004 --instance_task 1 --whichopt 0 \
     --init_type instance_resnet --cacheDirPrefix gs://cx_visualmaster/ --namefunc tpu_combine_tfutils_general \
     --tpu_flag 1 --tpu_name cx-cbnet-14 --tpu_task instance_task \
     --inst_lbl_pkl ../other_dataset/imagenet/all_label.pkl --instance_k 8192
"
for step in $(seq 1 50)
do
    ${test_str} --valid_first 0
    ${test_str} --valid_first 1
done
