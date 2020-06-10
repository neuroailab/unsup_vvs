# Neural fitting scripts

## Brain-score pipelines
Typical command: 
```
python -W ignore brainscore_mask/bs_fit_neural.py --gpu [gpu_number] --set_func [func_name] --bench_func [bench_func_name]
```
See `brainscore_mask/run_v1_v4_it.sh` for examples of doing v1, v4, and it fitting.
`brainscore_mask/run_single_behavior.sh` for i2n metrics.
`brainscore_mask/run_regression.sh` for regression metrics (translation, rotation).

## Visualization codes.
See folder `analyze_scripts`.

### Tensorflow 
Install lucid through `pip install --quiet lucid==0.2.3`.
Typical command:
```
python -W ignore analyze_scripts/lucid_stimuli_compute.py --gpu [gpu_number] --set_func [set_func]
```

### Pytorch
Install [torchsample](https://github.com/neuroailab/torchsample.git).
Typical command:
```
python -W ignore analyze_scripts/raw_pt_stimuli_compute.py --gpu [gpu_number] --set_func [set_func]
```
Or check `analyze_scripts/run_raw_pt_stimuli_compute.sh` for a layer-and-batch-separated call.

## Old pipelines
Old pipelines using `fit_neural_data.py` have been deprecated.
