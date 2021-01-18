Unsupervised deep convolutional neural network model for the ventral visual stream.

# Prerequisites

We use python=3.7.4.

`tensorflow_gpu==1.15.0`, tensorflow whose version is bigger than or equal to `1.9.0` and smaller than `2.0.0` should suffice for most of our network training. However, `SimCLR` training requires `tensorflow==1.15.0`.

# Instructions

The reproduction is done in three steps: preparing datasets needed for training, training networks, evaluating the trained networks using neural data.

To prepare datasets, see scripts in `prepare_datasets`.
To run the following two steps, you need to first install this package (like `pip install -e ./`).
For neural network training, see scripts in `unsup_vvs/network_training` folder.  
For neural data evaluation, see scripts in `unsup_vvs/neural_fit` folder.
