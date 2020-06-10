part=$2
for seed in 0 1 2
do
    sh brainscore_mask/run_single_behavior.sh $1 mt_settings.mt_${part}_s${seed}
done
