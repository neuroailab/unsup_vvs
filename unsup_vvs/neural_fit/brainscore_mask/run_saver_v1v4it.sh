gpu=${1}
set_file_name=${2}
set_name=${3}
param_id=${4}

for seed in seed0 seed1 seed2
do
    set_func=${set_file_name}_settings.${set_name}_${seed}
    python -W ignore brainscore_mask/bs_fit_neural.py \
        --gpu ${gpu} --set_func ${set_func} \
        --bench_func cadena_v1_mask_param_select_scores \
        --param_score_id ${param_id}-v1_range_norm \
        --id_suffix saver
    python -W ignore brainscore_mask/bs_fit_neural.py \
        --gpu ${gpu} --set_func ${set_func} \
        --param_score_id ${param_id} \
        --bench_func hvm_v4_mask_param_select_scores \
        --id_suffix saver
    python -W ignore brainscore_mask/bs_fit_neural.py \
        --gpu ${gpu} --set_func ${set_func} \
        --param_score_id ${param_id} \
        --bench_func hvm_it_mask_param_select_scores \
        --id_suffix saver
done
