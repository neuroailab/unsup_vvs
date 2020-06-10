#for part in p01 p02 p04 p06 p20 p50 p03 p05 p10
for part in p03 p05
do
    for seed in 0 1 2
    do
        sh brainscore_mask/run_single_behavior.sh $1 part_IN_settings.${part}_seed${seed}
    done
done
