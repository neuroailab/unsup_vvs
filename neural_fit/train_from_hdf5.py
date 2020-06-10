from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import cPickle
import pdb

import json
import copy
import argparse
import h5py
import time

from scipy.stats import pearsonr as pearson_r

# Loss function
def compute_corr(x, y):
    """
    Computes correlation of input vectors, using tf operations
    """
    x = tf.cast(tf.reshape(x, shape=(-1,)), tf.float32) # reshape and cast to ensure compatibility
    y = tf.cast(tf.reshape(y, shape=(-1,)), tf.float32)
    assert x.get_shape() == y.get_shape(), (x.get_shape(), y.get_shape())
    n = tf.cast(tf.shape(x)[0], tf.float32)
    x_sum = tf.reduce_sum(x)
    y_sum = tf.reduce_sum(y)
    xy_sum = tf.reduce_sum(tf.multiply(x, y))
    x2_sum = tf.reduce_sum(tf.pow(x, 2))
    y2_sum = tf.reduce_sum(tf.pow(y, 2))
    numerator = tf.scalar_mul(n, xy_sum) - tf.scalar_mul(x_sum, y_sum)
    denominator = tf.sqrt(tf.scalar_mul(tf.scalar_mul(n, x2_sum) - tf.pow(x_sum, 2),
				       tf.scalar_mul(n, y2_sum) - tf.pow(y_sum, 2))) + tf.constant(1e-4)
    corr = tf.truediv(numerator, denominator)

    return corr

def l2_and_corr(x, y):
    return tf.nn.l2_loss(x - y)/np.prod(y.get_shape().as_list()) - compute_corr(x, y)

# Function that build the linear mapping
def spa_disen_fc(
        out_shape,
        in_layer,
        weight_decay=None,
        weight_decay_type='l2',
        bias=0,
        trainable=True,
        seed=0):

    if weight_decay is None:
        weight_decay = 0.
    weight_decay_func = tf.contrib.layers.l2_regularizer
    if weight_decay_type == 'l1':
        weight_decay_func = tf.contrib.layers.l1_regularizer

    curr_shape = in_layer.get_shape().as_list()
    resh = tf.reshape(in_layer,
                      [curr_shape[0], -1],
                      name='reshape')

    # Needs to be outputs from convolution
    assert len(curr_shape) == 4, 'Need to be output from convolution!'
    spa_shape_x = curr_shape[1]
    spa_shape_y = curr_shape[2]
    cha_shape = curr_shape[3]

    spa_kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                                 shape=[spa_shape_x, spa_shape_y, 1, out_shape],
                                 dtype=tf.float32,
                                 regularizer=weight_decay_func(weight_decay),
                                 name='spa_weights', trainable=trainable)
    cha_kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                                 shape=[1, 1, cha_shape, out_shape],
                                 dtype=tf.float32,
                                 regularizer=weight_decay_func(weight_decay),
                                 name='cha_weights', trainable=trainable)
    kernel = spa_kernel * cha_kernel
    biases = tf.get_variable(initializer=tf.constant_initializer(bias),
                             shape=[out_shape],
                             dtype=tf.float32,
                             regularizer=weight_decay_func(weight_decay),
                             name='bias', trainable=trainable)

    kernel = tf.reshape(kernel, [-1, out_shape], name='ker_reshape')

    fcm = tf.matmul(resh, kernel)

    output = tf.nn.bias_add(fcm, biases, name='spa_disen_fc')

    return output

# Argument paser
def get_parser():
    parser = argparse.ArgumentParser(description='The script to fit to neural data using hdf5 data')

    # General setting
    parser.add_argument('--gpu', default='0', type=str, action='store',
	    help='Availabel GPUs')
    parser.add_argument('--train_path', 
            default='/data2/chengxuz/vm_response/response_resnet34_again.hdf5',
            type=str, action='store',
	    help='Path for the training hdf5')
    parser.add_argument('--val_path', 
            default='/data2/chengxuz/vm_response/response_resnet34_val_again.hdf5', 
            type=str, action='store',
	    help='Path for the validation hdf5')
    parser.add_argument('--fit_key', default='encode_5', type=str, action='store',
	    help='Key for network features')
    parser.add_argument('--label_key', default='V4_ave', type=str, action='store',
	    help='Key for label')

    parser.add_argument('--steps', default=15000, type=int, action='store',
		help='Number of steps')
    parser.add_argument('--val_steps', default=240, type=int, action='store',
		help='Number of training steps for one validation')
    parser.add_argument('--batchsize', default=64, type=int, action='store',
		help='Batch size')
    parser.add_argument('--seed', default=0, type=int, action='store',
		help='Random seed for model')
    parser.add_argument('--weight_decay', default=1e-3, type=float, action='store',
		help='Weight decay')
    parser.add_argument('--weight_decay_type', default='l2', type=str, action='store',
		help='Weight decay type, l2 (default) or l1')

    # Optimizer setting
    parser.add_argument('--init_lr', default=.01, type=float, action='store',
		help='Init learning rate')
    parser.add_argument('--adameps', default=0.1, type=float, action='store',
		help='Epsilon for adam')
    parser.add_argument('--adambeta1', default=0.9, type=float, action='store',
		help='Beta1 for adam')
    parser.add_argument('--adambeta2', default=0.999, type=float, action='store',
		help='Beta2 for adam')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Get all input data
    ## For training
    fin = h5py.File(args.train_path, 'r')
    all_train_data = fin[args.fit_key][:]
    all_train_label = fin[args.label_key][:]
    ## For testing
    fin_val = h5py.File(args.val_path, 'r')
    all_val_data = fin_val[args.fit_key][:]
    all_val_label = fin_val[args.label_key][:]

    train_len = all_train_data.shape[0]
    val_len = all_val_data.shape[0]
    assert (train_len % args.batchsize==0) & (val_len % args.batchsize==0), "Should set a good batchsize!"

    # Build the graph
    ## Get placeholder
    network_input = tf.placeholder(tf.float32, [args.batchsize] + list(all_train_data.shape[1:]))
    label_input = tf.placeholder(tf.float32, [args.batchsize] + list(all_train_label.shape[1:]))

    ## Build the network
    m_output = spa_disen_fc(
            out_shape=all_train_label.shape[1],
            in_layer=network_input,
            weight_decay=args.weight_decay,
            weight_decay_type=args.weight_decay_type,
            seed=args.seed)

    ## Build the loss
    pure_loss = l2_and_corr(m_output, label_input)

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    final_loss = tf.add(tf.add_n(reg_losses), pure_loss)

    ## Build the optimizer
    optimizer = tf.train.AdamOptimizer(
            learning_rate=args.init_lr,
            epsilon=args.adameps, 
            beta1=args.adambeta1,
            beta2=args.adambeta2,)
    opt_op = optimizer.minimize(final_loss)

    # Create tensorflow session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init_op_global = tf.global_variables_initializer()
    sess.run(init_op_global)
    init_op_local = tf.local_variables_initializer()
    sess.run(init_op_local)

    # Do the training
    for idx_step in xrange(args.steps):
        ## Get current training data
        data_idx_start = (idx_step*args.batchsize) % train_len
        data_idx_end = ((idx_step+1)*args.batchsize) % train_len
        if data_idx_end==0:
            data_idx_end = train_len
        curr_data = all_train_data[data_idx_start:data_idx_end]
        curr_label = all_train_label[data_idx_start:data_idx_end]

        # Do the training and report
        curr_pure_loss, curr_all_loss, _ = sess.run(
                [pure_loss, final_loss, opt_op], 
                feed_dict = {network_input: curr_data, label_input: curr_label})
        print('Step %i, pure loss: %.4f, all loss: %.4f' % (idx_step, curr_pure_loss, curr_all_loss))

        # Do the validation if needed
        if (idx_step+1)%args.val_steps==0:
            print('Do validation now')
            all_val_output = np.zeros(all_val_label.shape)
            for val_idx_step in xrange(int(val_len/args.batchsize)):
                val_idx_start = val_idx_step*args.batchsize
                val_idx_end = (val_idx_step+1)*args.batchsize

                curr_val_data = all_val_data[val_idx_start:val_idx_end]
                curr_m_output = sess.run(m_output, feed_dict={network_input:curr_val_data})
                all_val_output[val_idx_start:val_idx_end] = curr_m_output

            ## Compute the median of correlation
            all_pear_corr = []
            for idx_neuron in xrange(all_val_label.shape[1]):
                curr_pear_corr = pearson_r(all_val_label[:,idx_neuron], all_val_output[:,idx_neuron])[0]
                all_pear_corr.append(curr_pear_corr)
            print('Median of correlation value: %.4f' % np.median(all_pear_corr))

if __name__ == '__main__':
    main()
