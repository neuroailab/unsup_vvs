for seed in seed0 seed1
do
    method_prefix=color_settings.color_
    for region in V1 V4
    do
        python -W ignore analyze_scripts/stimuli_compute.py \
            --set_func ${method_prefix}${seed} \
            --gpu ${1} --region ${region} \
            --with_tv_jitter \
            --layers encode_1.conv,encode_1,encode_2,encode_3,encode_4,encode_6,encode_9
        python -W ignore analyze_scripts/stimuli_compute.py \
            --set_func ${method_prefix}${seed} \
            --gpu ${1} --region ${region} \
            --layers encode_5,encode_7,encode_8 \
            --with_tv_jitter \
            --tv_wd 1e-2 --wd 1e-2
    done
    region=IT
    python -W ignore analyze_scripts/stimuli_compute.py \
        --set_func ${method_prefix}${seed} \
        --gpu ${1} --region ${region} \
        --with_tv_jitter \
        --tv_wd 1e-3 --wd 1e-3
done
