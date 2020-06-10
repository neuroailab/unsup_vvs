for seed in seed0 seed1 seed2
do
    for method_prefix in cate_settings.cate_ la_settings.la_ color_settings.color_ dc_settings.dc_ untrained_settings.untrn_
    do
        python -W ignore analyze_scripts/circular_var.py --set_func ${method_prefix}${seed} --gpu ${1}
    done
done
