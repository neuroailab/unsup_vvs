# Prerequisites

First run `install_brainscore_specific_commits.sh` to install brainscore related repos at specific commits.
This is due to the fact that our current benchmark implementations are based on earlier commits of brainscore repos.
We are working actively to make our benchmark implementations runable with the latest version of brainscore repos.

Before starting, please modify the value of `DEFAULT_MODEL_CACHE_DIR` at line 17 of `brainscore_mask/tf_model_loader.py` and the value of `DEFAULT_RESULTCACHING_HOME` at line 14 of `brainscore_mask/bs_fit_neural.py`.
The first value is where the network model caches will be hosted.
The second value is where the neural evaluation results will be hosted.

To get neural fitting results to V1, V4, and IT areas, you need to run `sh brainscore_mask/run_v1_v4_it.sh [gpu_number] [set_func_name]` (Currently, certain credentials may be needed).

## Supervised

`set_func_name` should be `supervised.super_res18_s0`, `supervised.super_res18_s1`, `supervised.super_res18_s2`.

## Local Aggregation

`set_func_name` should be `la.res18_la_s0`, `la.res18_la_s1`, `la.res18_la_s2`.


## Instance Recognition

`set_func_name` should be `ir.res18_ir_s0`, `ir.res18_ir_s1`, `ir.res18_ir_s2`.


## Relative Position

`set_func_name` should be `rp.res18_rp_s0`, `rp.res18_rp_s1`, `rp.res18_rp_s2`.


## Colorization

`set_func_name` should be `col.res18_col_s0`, `col.res18_col_s1`, `col.res18_col_s2`.


## CPC

`set_func_name` should be `cpc.res18_cpc_s0`, `cpc.res18_cpc_s1`, `cpc.res18_cpc_s2`.


## Auto-Encoder

`set_func_name` should be `ae.res18_ae_s0`, `ae.res18_ae_s1`, `ae.res18_ae_s2`.
