# Scripts for network training

Typical command:
```
python train_combinet_clean.py --gpu [gpu_numbers] --cacheDirPrefix /mnt/fs4/chengxuz --load_setting_func [set_func_name]
```

`saved_setting.py` includes most of the ImageNet category settings, including res50, res18. 
Doing set_func in this file just needs the function name.

Other setting files inside folder `exp_settings` requires first having the file name and then the set name in the argument, sperated by `.`.
