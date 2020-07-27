gpu=$1
set_func=$2

for layer_start_idx in $(seq 0 9)
do
    for global_start_idx in $(seq 0 16 48)
    do
        python -W ignore analyze_scripts/raw_pt_stimuli_compute.py \
            --gpu ${gpu} --set_func ${set_func} \
            --global_start_idx ${global_start_idx} --num_batches 1 \
            --layer_start_idx ${layer_start_idx}
    done
done
