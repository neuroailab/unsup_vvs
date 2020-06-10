RESULTCACHING_DISABLE=model_tools.activations python -W ignore brainscore_mask/bs_fit_neural.py \
    --gpu ${1} --set_func ${2} \
    --bench_func cadena_v1_mask_param_select_scores \
    --id_suffix v1_range_norm_${3}
RESULTCACHING_DISABLE=model_tools.activations python -W ignore brainscore_mask/bs_fit_neural.py \
    --gpu ${1} --set_func ${2} \
    --bench_func hvm_v4_mask_param_select_scores --id_suffix ${3}
RESULTCACHING_DISABLE=model_tools.activations python -W ignore brainscore_mask/bs_fit_neural.py \
    --gpu ${1} --set_func ${2} \
    --bench_func hvm_it_mask_param_select_scores --id_suffix ${3}
