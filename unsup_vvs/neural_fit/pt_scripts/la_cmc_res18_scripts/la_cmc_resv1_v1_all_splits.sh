model_ckpt=/mnt/fs4/chengxuz/la_cmc_models/experiments/imagenet_la_cmc/res18_lab_cmc+la_v1_s0/checkpoints/final.pth.tar
for which_split in 0 1 2 3 4
do
    python main.py \
        --data /mnt/fs4/chengxuz/v1_cadena_related/tfrs/split_${which_split}/images \
        --save_path /mnt/fs4/chengxuz/v4it_temp_results/la_cmc_res18v1_nf/V1_split_${which_split} \
        --model_ckpt ${model_ckpt} \
        --dataset_type v1_tc --resv1
done
