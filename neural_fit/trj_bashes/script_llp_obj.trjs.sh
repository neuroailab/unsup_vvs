res_dir=/mnt/fs4/chengxuz/v4it_temp_results/llp_res18_obj

for k in $(seq 1 5)
do
    a=$(( 20018*k ))
    hdf5_path=${res_dir}/ckpt_${k}.hdf5
    python fit_neural_data.py --gpu 1 \
        --load_setting_func inst_ctl_model_obj_to_be_filled \
        --loadstep $a \
        --expId llp_obj_trj \
        --gen_hdf5path ${hdf5_path}
    python3.6 behavior_using_stream.py \
        --hdf5_path ${hdf5_path} \
        --save_path ${res_dir}/ckpt_${k}.pkl
done

for k in $(seq 10 5 75) $(seq 130 5 180)
do
    a=$(( 20018*k ))
    hdf5_path=${res_dir}/ckpt_${k}.hdf5
    python fit_neural_data.py --gpu 1 \
        --load_setting_func llp_p03_obj_to_be_filled \
        --loadstep $a \
        --expId llp_obj_trj \
        --gen_hdf5path ${hdf5_path}
    python3.6 behavior_using_stream.py \
        --hdf5_path ${hdf5_path} \
        --save_path ${res_dir}/ckpt_${k}.pkl
done
