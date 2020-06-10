python -W ignore brainscore_mask/bs_fit_neural.py \
    --gpu ${1} --set_func ${2} \
    --bench_func cadena_v1_mask_param_select_scores \
    --ls_ld_range v1_cadena_range --id_suffix v1_range_norm
