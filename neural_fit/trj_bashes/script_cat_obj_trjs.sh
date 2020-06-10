res_dir=/mnt/fs4/chengxuz/v4it_temp_results/cate_res18_obj

for k in $(seq 1 4) $(seq 5 5 64)
do
    a=$(( 10009*k ))
    hdf5_path=${res_dir}/ckpt_${k}.hdf5
    python fit_neural_data.py --gpu 0 \
        --load_setting_func cate_res18_obj_to_be_filled \
        --ckpt_file /mnt/fs4/chengxuz/tpu_ckpts/res18/model.ckpt-$a \
        --gen_hdf5path ${hdf5_path}
    python3.6 behavior_using_stream.py \
        --hdf5_path ${hdf5_path} \
        --save_path ${res_dir}/ckpt_${k}.pkl
done
