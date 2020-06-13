# Prerequisites

Our network training is based on tfutils. 
Please install [tfutils](https://github.com/neuroailab/tfutils) first.
To use tfutils, you need to host a mongodb, please see general instructions about how to do that.
We suggest making the mongodb accessible through port 27007, so you don't need to modify experiment settings.
Due to a conflict in requirements which will not influence this repo, you need to reinstall `tensorflow_gpu==1.15.0` after installing tfutils.

# Scripts for network training

Typical command:
```
python train_combinet_clean.py --gpu [gpu_numbers] --cacheDirPrefix /path/to/model/cache --load_setting_func [set_func_name] --overall_local /path/to/datasets
```

This `overall_local` parameter is used to set the dataset path. 
Please check `utilities/data_path_utils.py` for details. 
If you have set the dataset directories as suggested when preparing the datasets, this is all you need. 
Otherwise, you need to modify contents in `utilities/data_path_utils.py`.

The `set_func_name` parameter should be `script_name.function_name`. Here the `script_name` should be scripts hosted in `exp_settings` folder. `function_name` is the function in the script you want to run.

## Supervised

`set_func_name` should be `supervised.cate_res18_exp0`, `supervised.cate_res18_exp1`, `supervised.cate_res18_exp2`, corresponding to three networks with different initializations.

## Local Aggregation

`set_func_name` should be `la.res18_la_s0`, `la.res18_la_s1`, `la.res18_la_s2`.


## SimCLR

SimCLR needs to be trained through preparing ImageNet following instructions in folder `models/simclr` and using TPUs (v3-pod-256 recommended). After preparing the dataset and building TPUs, run scripts `training_bashes/train_res18_in_0.sh`, `training_bashes/train_res18_in_1.sh`, and `training_bashes/train_res18_in_2.sh` in that folder.


## Instance Recognition

`set_func_name` should be `ir.res18_ir_s0`, `ir.res18_ir_s1`, `ir.res18_ir_s2`.
