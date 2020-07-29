#for part in p02
#for part in p04
#for part in p06
#for part in p20
#for part in p50
part=$2
for seed in 1 2
do
    sh brainscore_mask/run_single_behavior.sh $1 llp_settings.llp_${part}_seed${seed}
done
sh brainscore_mask/run_single_behavior.sh $1 llp_settings.llp_${part}
