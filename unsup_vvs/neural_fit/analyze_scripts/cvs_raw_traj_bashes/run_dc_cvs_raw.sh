python -W ignore analyze_scripts/circular_var_raw.py --set_func dc_settings.dc_seed1 --gpu ${1}
method_prefix=dc_settings.dc_seed1_
for traj_ep in ep2 ep4 ep6 ep8 ep10 ep20 ep40 ep70
do
    python -W ignore analyze_scripts/circular_var_raw.py --set_func ${method_prefix}${traj_ep} --gpu ${1}
done
