python -W ignore analyze_scripts/circular_var_raw.py --set_func color_settings.color_seed0 --gpu ${1}
method_prefix=color_settings.color_seed0_
for traj_ep in ep2 ep4 ep6 ep8 ep10 ep20 ep30 ep40 ep50 ep100 ep150 ep200
do
    python -W ignore analyze_scripts/circular_var_raw.py --set_func ${method_prefix}${traj_ep} --gpu ${1}
done
