model_ckpt=/mnt/fs6/honglinc/trained_models/res18_Lab_cmc+la/checkpoints/checkpoint_epoch190.pth.tar

python main.py \
    --save_path /mnt/fs4/chengxuz/v4it_temp_results/la_cmc_res18_nf/V4IT_split_0 \
    --model_ckpt ${model_ckpt}
python main.py \
    --data /mnt/fs0/datasets/neural_data/img_split/V4IT_split_2/tf_records/images \
    --save_path /mnt/fs4/chengxuz/v4it_temp_results/la_cmc_res18_nf/V4IT_split_1 \
    --model_ckpt ${model_ckpt}
python main.py \
    --data /mnt/fs0/datasets/neural_data/img_split/V4IT_split_3/tf_records/images \
    --save_path /mnt/fs4/chengxuz/v4it_temp_results/la_cmc_res18_nf/V4IT_split_2 \
    --model_ckpt ${model_ckpt}
python main.py \
    --data /mnt/fs0/datasets/neural_data/img_split/V4IT_split_4/tf_records/images \
    --save_path /mnt/fs4/chengxuz/v4it_temp_results/la_cmc_res18_nf/V4IT_split_3 \
    --model_ckpt ${model_ckpt}
python main.py \
    --data /mnt/fs0/datasets/neural_data/img_split/V4IT_split_5/tf_records/images \
    --save_path /mnt/fs4/chengxuz/v4it_temp_results/la_cmc_res18_nf/V4IT_split_4 \
    --model_ckpt ${model_ckpt}
