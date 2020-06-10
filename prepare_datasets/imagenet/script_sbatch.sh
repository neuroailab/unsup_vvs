#lenindx=5
lenindx=10
#lenindx=2

for staindx in $(seq 0 ${lenindx} 1024)
#for staindx in $(seq 0 ${lenindx} 50)
#for staindx in 0
do
    #sbatch script_to_tfrs.sh ${staindx} ${lenindx}
    sbatch script_combine.sh ${staindx} ${lenindx}
done
