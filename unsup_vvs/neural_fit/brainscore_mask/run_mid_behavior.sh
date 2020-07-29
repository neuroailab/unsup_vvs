#for method in la ir ae untrn
#for method in depth color rp
#for method in cpc
#for method in simclr
for method in dc cmc 
do
    for seed in 0 1 2
    do
        sh brainscore_mask/run_single_behavior.sh $1 i2nmid_settings.${method}_seed${seed}
    done
done
