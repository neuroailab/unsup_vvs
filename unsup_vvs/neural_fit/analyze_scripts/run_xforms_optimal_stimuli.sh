for seed in seed0 seed1 seed2
do
    for method_prefix in cate_settings.cate_ la_settings.la_ color_settings.color_ untrained_settings.untrn_
    do
        python -W ignore analyze_scripts/stimuli_compute.py --set_func ${method_prefix}${seed} --gpu ${1} --with_xforms
        python -W ignore analyze_scripts/stimuli_compute.py --set_func ${method_prefix}${seed} --gpu ${1} --region V1 --with_xforms
        python -W ignore analyze_scripts/stimuli_compute.py --set_func ${method_prefix}${seed} --gpu ${1} --region IT --with_xforms
    done
done
