method_prefix=${2}
for seed in seed0 seed1
do
    for region in V4 IT V1
    do
        for wd in 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5
        do
            python -W ignore analyze_scripts/stimuli_compute.py \
                --set_func ${method_prefix}${seed} \
                --gpu ${1} --region ${region} --tv_wd ${wd} --wd ${wd} \
                --with_tv_jitter \
                --special _tv_jitter_${wd}
        done
    done
done
