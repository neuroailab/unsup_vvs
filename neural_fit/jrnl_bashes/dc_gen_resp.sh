cd ~/deepcluster

python output_hvm.py --data /mnt/fs0/datasets/neural_data/img_split/V4IT/tf_records/images \
    --model /mnt/fs4/chengxuz/deepcluster_models/res18_np/checkpoint.pth.tar \
    --conv 104,105,106,107,108,109,110,111,112 \
    --save_path /mnt/fs4/chengxuz/v4it_temp_results/deepcluster_hvm/res18_np/V4IT_split_0 --resnet_deseq

python output_hvm.py --data /mnt/fs0/datasets/neural_data/img_split/V4IT_split_2/tf_records/images \
    --model /mnt/fs4/chengxuz/deepcluster_models/res18_np/checkpoint.pth.tar \
    --conv 104,105,106,107,108,109,110,111,112 \
    --save_path /mnt/fs4/chengxuz/v4it_temp_results/deepcluster_hvm/res18_np/V4IT_split_1 --resnet_deseq

python output_hvm.py --data /mnt/fs0/datasets/neural_data/img_split/V4IT_split_3/tf_records/images \
    --model /mnt/fs4/chengxuz/deepcluster_models/res18_np/checkpoint.pth.tar \
    --conv 104,105,106,107,108,109,110,111,112 \
    --save_path /mnt/fs4/chengxuz/v4it_temp_results/deepcluster_hvm/res18_np/V4IT_split_2 --resnet_deseq

python output_hvm.py --data /mnt/fs0/datasets/neural_data/img_split/V4IT_split_4/tf_records/images \
    --model /mnt/fs4/chengxuz/deepcluster_models/res18_np/checkpoint.pth.tar \
    --conv 104,105,106,107,108,109,110,111,112 \
    --save_path /mnt/fs4/chengxuz/v4it_temp_results/deepcluster_hvm/res18_np/V4IT_split_3 --resnet_deseq

python output_hvm.py --data /mnt/fs0/datasets/neural_data/img_split/V4IT_split_5/tf_records/images \
    --model /mnt/fs4/chengxuz/deepcluster_models/res18_np/checkpoint.pth.tar \
    --conv 104,105,106,107,108,109,110,111,112 \
    --save_path /mnt/fs4/chengxuz/v4it_temp_results/deepcluster_hvm/res18_np/V4IT_split_4 --resnet_deseq
