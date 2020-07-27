for step in $(seq 0 60)
do
    python train_combinet.py "$@" --valid_first 0
    python train_combinet.py "$@" --valid_first 1
done
