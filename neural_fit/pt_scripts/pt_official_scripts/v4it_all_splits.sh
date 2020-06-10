data_prefix=/mnt/fs0/datasets/neural_data/img_split/V4IT_split_
save_prefix=/mnt/fs4/chengxuz/v4it_temp_results/pt_official_res18_nf/V4IT_split_

python main_pt_official.py
for which_split in 1 2 3 4
do
    data_split=$((which_split + 1))
    python main_pt_official.py \
        --data ${data_prefix}${data_split}/tf_records/images \
        --save_path ${save_prefix}${which_split}
done
