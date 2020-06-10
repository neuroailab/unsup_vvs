now_len=10
for now_start in $(seq 1 ${now_len} 163)
#for now_start in $(seq 153 158)
#for now_start in $(seq 130 132)
#for now_start in 153 156 157 158
do
    python download_synset_jpgs.py --syn_sta ${now_start} --syn_len ${now_len}&
done
