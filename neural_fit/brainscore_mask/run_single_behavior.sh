set_func=${2}
RESULTCACHING_DISABLE=model_tools.activations python -W ignore \
    brainscore_mask/bs_fit_neural.py \
    --set_func ${set_func} \
    --bench_func objectome_i2n_with_save_layer_param_scores \
    --gpu ${1} --id_suffix save
