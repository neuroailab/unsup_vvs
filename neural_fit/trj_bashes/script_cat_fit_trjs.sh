for k in $(seq 1 4) $(seq 5 5 64)
do
    a=$(( 10009*k ))
    a_end=$(( a+20000 ))
    python fit_neural_data.py --gpu 6 \
        --load_setting_func cate_res18_to_be_filled \
        --expId nf_trj_cate_res18_$k \
        --ckpt_file /mnt/fs4/chengxuz/tpu_ckpts/res18/model.ckpt-$a \
        --train_num_steps ${a_end}
done
