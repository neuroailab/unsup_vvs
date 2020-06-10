# instance_task
Reimplement and improve the self-supervised task in "Unsupervised Feature Learning via Non-Parametric Instance Discrimination"

# Instructions for training

1. Install the `master` branch of `tfutils`.
```
git clone https://github.com/neuroailab/tfutils.git
cd tfutils
git checkout master
python setup.py install --user
```

2. Run script
```
python train_tfutils.py --config ./exp_configs/control.json:small --gpu [your gpu number]
```
You can inspect for `control.json` to see some of the configuration options.

Besides, using gpu numbers separated by `,` will start multi-gpu training, when batch size is the overall batch size for all gpus.

Depending on configuration, it may be necessary to do port forwarding to another machine that hosts mongodb, e.g.
```
ssh -fNL 27009:localhost:27009 [user]@node11-neuroaicluster
```
