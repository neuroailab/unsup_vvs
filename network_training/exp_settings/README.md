# Files containing different set funcs

## combine_irla_others.py

Functions about ir, la, and combinations of ir and la with other tasks.

## imagenet_transfer.py

All the ImageNet transfer settings used in the paper. Pytorch transfer learning is done through `neural_fit/analyze_scripts/pt_imagenet_transfer.py`.

## other_tasks.py

Including depth, rp, colorization, ae, cpc tasks. 
However, the ckpts actually used for depth, rp, and colorization tasks are trained through TPU bashes stored in `tpu` folder (under `combine_pred` folder).
