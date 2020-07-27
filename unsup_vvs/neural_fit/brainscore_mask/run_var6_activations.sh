python -W ignore brainscore_mask/bs_fit_neural.py \
    --set_func ${2}_settings.${3}_seed0 \
    --gpu ${1} --bench_func hvm_var6_activations
python -W ignore brainscore_mask/bs_fit_neural.py \
    --set_func ${2}_settings.${3}_seed1 \
    --gpu ${1} --bench_func hvm_var6_activations
python -W ignore brainscore_mask/bs_fit_neural.py \
    --set_func ${2}_settings.${3}_seed2 \
    --gpu ${1} --bench_func hvm_var6_activations
