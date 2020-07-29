:"
for set_func in \
    cate_settings.cate_seed0 \
    cate_settings.cate_p03 \
    llp_settings.llp_p03 \
    mt_settings.mt_p03 \
    untrained_settings.untrn_seed0 \
    la_settings.old_la_mid \
    la_settings.la_seed0 \
    ir_settings.ir_seed0 \
    color_settings.color_seed0 \
    rp_settings.rp_seed0 \
    depth_settings.depth_seed0 \
    ae_settings.ae_seed0 \
    dc_settings.dc_seed0 \
    la_cmc_settings.cmc_seed0 \
    cpc_settings.cpc_seed0
do
    RESULTCACHING_DISABLE=model_tools.activations python -W ignore \
        brainscore_mask/bs_fit_neural.py \
        --set_func ${set_func} \
        --bench_func objectome_i2n_with_save_layer_param_scores \
        --gpu ${1} --id_suffix save
    #RESULTCACHING_DISABLE=model_tools.activations python -W ignore \
    #    brainscore_mask/bs_fit_neural.py \
    #    --set_func ${set_func} \
    #    --bench_func objectome_i2n_with_save_spearman_layer_param_scores \
    #    --gpu ${1} --id_suffix save
done
"

#for method in cate la ir ae untrn simclr depth color rp cpc cmc dc
for method in depth color rp cmc dc
do
    for seed in 0 1 2
    do
        sh brainscore_mask/run_single_behavior.sh $1 ${method}_settings.${method}_seed${seed}
    done
done
