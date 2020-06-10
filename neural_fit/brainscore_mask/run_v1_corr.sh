RESULTCACHING_DISABLE=model_tools.activations python -W ignore brainscore_mask/bs_fit_neural.py \
    --gpu ${1} --set_func ${2} \
    --bench_func cadena_v1_correlation_scores \
    --id_suffix corr
