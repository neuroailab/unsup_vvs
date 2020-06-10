set_func=$2
id_suffix=cate2

python -W ignore \
    brainscore_mask/bs_fit_neural.py \
    --set_func ${set_func} \
    --bench_func pca_category_layer_param_select_scores \
    --gpu ${1} --id_suffix ${id_suffix}
