method_prefix=${2}
for seed in seed0 seed1
do
    for wd in 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5
    do
        python -W ignore analyze_scripts/raw_stimuli_compute.py \
            --set_func ${method_prefix}${seed} \
            --gpu ${1} --tv_wd ${wd} --wd ${wd} \
            --special _${wd}
    done
done
