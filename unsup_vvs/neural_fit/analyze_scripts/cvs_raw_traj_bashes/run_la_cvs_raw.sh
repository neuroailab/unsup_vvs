python -W ignore analyze_scripts/circular_var_raw.py --set_func la_settings.la_seed1 --gpu ${1}
method_prefix=la_settings.la_seed1_
for traj_ep in ep2 ep4 ep6 ep8 ep10 ep20 ep40 ep80 ep120 ep160 ep180 ep220 ep240
do
    python -W ignore analyze_scripts/circular_var_raw.py --set_func ${method_prefix}${traj_ep} --gpu ${1}
done
