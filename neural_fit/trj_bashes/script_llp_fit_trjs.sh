for k in $(seq 1 5)
do
    a=$(( 20018*k ))
    a_end=$(( a+20000 ))
    python fit_neural_data.py --gpu 7 \
        --load_setting_func inst_ctl_model_to_be_filled \
        --expId nf_trj_inst_ctl_$k \
        --loadstep $a \
        --train_num_steps ${a_end}
done

for k in $(seq 10 5 75) $(seq 130 5 180)
do
    a=$(( 20018*k ))
    a_end=$(( a+20000 ))
    python fit_neural_data.py --gpu 7 \
        --load_setting_func llp_p03_to_be_filled \
        --expId nf_trj_llp_p03_$k \
        --loadstep $a \
        --train_num_steps ${a_end}
done
