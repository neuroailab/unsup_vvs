python -W ignore brainscore_mask/bs_fit_neural.py \
    --gpu ${1} --set_func ${2} \
    --bench_func cadena_v1_mask_param_select_scores \
    --layer_separate \
    --ls_ld_range v1_cadena_input_range --id_suffix v1_input_range_norm \
    --just_input
python -W ignore brainscore_mask/bs_fit_neural.py \
    --gpu ${1} --set_func ${2} \
    --bench_func cadena_v1_mask_param_select_scores \
    --layer_separate \
    --ls_ld_range v1_cadena_range --id_suffix v1_range_norm \
    --just_input
python -W ignore brainscore_mask/bs_fit_neural.py \
    --gpu ${1} --set_func ${2} \
    --layer_separate \
    --bench_func hvm_v4_mask_param_select_scores \
    --just_input
python -W ignore brainscore_mask/bs_fit_neural.py \
    --gpu ${1} --set_func ${2} \
    --layer_separate \
    --bench_func hvm_it_mask_param_select_scores \
    --just_input
