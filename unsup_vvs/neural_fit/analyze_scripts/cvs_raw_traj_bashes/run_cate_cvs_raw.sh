method_prefix=cate_settings.cate_seed0_
for traj_ep in ep02 ep04 ep06 ep08 ep1 ep2 ep3 ep4 ep5 ep10 ep20 ep35 ep55 ep65 ep85
do
    python -W ignore analyze_scripts/circular_var_raw.py --set_func ${method_prefix}${traj_ep} --gpu ${1}
done
