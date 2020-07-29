from __future__ import division, print_function, absolute_import
import os
import sys

import numpy as np
import tensorflow as tf
import argparse
from collections import OrderedDict
import copy
import pdb

from tfutils import base, optimizer
from tfutils.utils import online_agg
from tfutils.defaults import mean_and_reg_loss

CONVRNN_REPO_PATH = os.path.expanduser('~/ntcnet/convrnn_batch_norm_training')
sys.path.append(CONVRNN_REPO_PATH)
sys.path.append(os.path.join(CONVRNN_REPO_PATH, '../decoder_search/'))
from imagenet_dataprovider_gpu import ImageNet 
from tnn_base_edges import tnn_base as tnn_base_fb

from gpu_train_script import get_parser, get_params_from_arg
slim = tf.contrib.slim

edges_2 = [(('conv8', 'conv5'), 0.0), (('conv9', 'conv6'), 0.0)]
edges_3 = edges_2 + [(('conv10', 'conv7'), 0.0)]
edges_5 = edges_3 + [(('conv7', 'conv6'), 0.0), (('conv10', 'conv9'), 0.0)]


def train_median_wfb(load_nm='medfb_configmodel438e5t0.npz',
                     cell_layers=['conv' + str(i) for i in range(4, 11)],
                     edges_arr=[],
                     input_args=[]):

    # Parse arguments
    parser = get_parser()
    args = parser.parse_args(input_args)

    # Get params needed, start training
    print(args)
    default_train_params = get_params_from_arg(args)

    params_copy = copy.copy(default_train_params)

    train_config = np.load(os.path.join(CONVRNN_REPO_PATH, 'model_config_npzs/', load_nm))['arr_0'][()]

    print('Loading config: ', train_config)

    model_params = train_config['model_params']

    cell_params = model_params.pop('cell_params')
    # add fb specific params
    cell_params['feedback_activation'] = tf.identity
    cell_params['feedback_entry'] = 'out'
    cell_params['feedback_depth_separable'] = False
    cell_params['feedback_filter_size'] = 1

    if args.layer_norm is not None:
        print('Setting layer norm to ', args.layer_norm)
        cell_params['layer_norm'] = args.layer_norm

    layer_params = {
            'conv1':{'cell_params': None}, 
            'conv2':{'cell_params':None}, 
            'conv3':{'cell_params':None}, 
            'conv4':{'cell_params':None}, 
            'conv5':{'cell_params':None}, 
            'conv6':{'cell_params':None}, 
            'conv7':{'cell_params':None}, 
            'conv8':{'cell_params':None}, 
            'conv9':{'cell_params':None}, 
            'conv10':{'cell_params':None}, 
            'imnetds':{'cell_params':None}}
    for k in cell_layers:
        print(k, 'layer norm', cell_params['layer_norm'])
        layer_params[k]['cell_params'] = cell_params

    model_params['layer_params'] = layer_params
    if args.times is not None:
        model_params['times'] = args.times
        print('Times now set to ', model_params['times'])

    if args.ff_bn:
        print('Setting ff batch norm to be true')
        bn_nodes_val = ['conv' + str(i) for i in range(1, 11)]
        model_params['bn_nodes'] = bn_nodes_val

    model_params['batch_norm_decay'] = args.batch_norm_decay
    model_params['batch_norm_epsilon'] = args.batch_norm_epsilon

    model_params['unroll_tf'] = args.unroll_tf
    if args.const_pres is not None:
        model_params['const_pres'] = args.const_pres

    if args.rnn_bn is not None:
        print('ARGS RNN bn', args.rnn_bn)
        model_params['rnn_bn'] = args.rnn_bn

    if args.rnn_bn_inp is not None:
        print('ARGS RNN bn inp', args.rnn_bn_inp)
        model_params['rnn_bn_inp'] = args.rnn_bn_inp

    if args.rnn_bn_all_convs is not None:
        print('ARGS RNN bn all convs', args.rnn_bn_all_convs)
        model_params['rnn_bn_all_convs'] = args.rnn_bn_all_convs
        model_params['edges_init_zero'] = args.edges_init_zero

    if args.batch_norm_cell_out is not None:
        print('ARGS RNN batch norm cell out', args.batch_norm_cell_out)
        model_params['batch_norm_cell_out'] = args.batch_norm_cell_out

    if args.gate_tau_bn_gamma_init is not None:
        print('ARGS RNN gate_tau_bn_gamma_init', args.gate_tau_bn_gamma_init)
        model_params['gate_tau_bn_gamma_init'] = args.gate_tau_bn_gamma_init


    lr_params = train_config['learning_rate_params']
    optimizer_params = train_config.get('optimizer_params')

    model_params['ff_weight_decay'] = None # use the values specified in json
    model_params['ff_kernel_initializer_kwargs'] = None # use the values specified in json
    model_params['final_max_pool'] = True
    if args.decoder_type == 'last':
        print('Using default last timestep decoder')
        model_params['decoder_end'] = model_params['times']
        model_params['decoder_start'] = model_params['decoder_end'] - 1
        model_params['decoder_type'] = 'last'
    else:
        print('Using ', args.decoder_type, ' decoder')
        model_params['decoder_end'] = model_params['times']
        model_params['decoder_start'] = 11 # first timestep of feedforward output
        model_params['decoder_type'] = args.decoder_type
        model_params['decoder_trainable'] = args.decoder_trainable

    # apply model params
    for k in model_params.keys():
        params_copy['model_params'][k] = model_params[k]

    params_copy['model_params']['func'] = tnn_base_fb
    params_copy['model_params']['base_name'] = args.tnn_json
    params_copy['model_params']['convrnn_type'] = 'recipcell'

    if args.use_edges:
        print('Passing in edges', edges_arr)
        params_copy['model_params']['edges_arr'] = edges_arr
    else:
        print('IGNORING any edges passed in')
        params_copy['model_params']['edges_arr'] = []

    if args.use_resnet:
        print('Using Resnet')
        params_copy['model_params'] = {'func': resnet_func, 'base_name': args.tnn_json, 'unroll_tf': args.unroll_tf}
        if args.times is not None:
           params_copy['model_params']['ntimes'] = args.times

    if args.use_google_resnet:
        print('Using Google Resnet')
        params_copy['model_params'] = {'func': google_resnet_func, 'resnet_size': args.resnet_size}

    init_lr_val = lr_params['learning_rate']
    if args.rescale_lr:
        print('Rescaling init lr')
        init_lr_val = (init_lr_val) * (args.train_batch_size / 128.0)
    if args.init_lr_fac10:
        print('Multiplying init lr by factor of 10')
        init_lr_val = init_lr_val * 10

    if args.use_resnet or args.use_google_resnet:
        init_lr_val = 0.1 * (args.train_batch_size / 256.0)

    if args.do_restore is not None:
        params_copy['load_params']['do_restore'] = True
        if args.load_step is not None:
            params_copy['load_params']['query'] = {'step': args.load_step}
        else:
            params_copy['load_params']['query'] = None

    if args.drop == 0:
        if (not args.use_resnet) and (not args.use_google_resnet):
            for k in lr_params.keys():
                params_copy['learning_rate_params'][k] = lr_params[k]
        params_copy['learning_rate_params']['learning_rate'] = init_lr_val
    elif args.drop > 0:
        prev_lr = ((0.1)**(args.drop-1))*(init_lr_val)
        next_lr = ((0.1)**args.drop)*(init_lr_val)
        print('Prev_lr:', prev_lr, 'next_lr:', next_lr, 'boundary_step:', args.boundary_step)
        params_copy['learning_rate_params'] = {'func': drop_learning_rate, 'boundary_step': args.boundary_step, 'next_lr': next_lr, 'prev_lr': prev_lr}

        params_copy['load_params']['do_restore'] = True
        if args.load_step is not None:
            params_copy['load_params']['query'] = {'step': args.load_step}
        else:
            params_copy['load_params']['query'] = None
    
    if args.load_exp is not None:
        params_copy['load_params']['exp_id'] = args.load_exp

    if (optimizer_params is not None) and (not args.use_resnet) and (not args.use_google_resnet):
        for k in optimizer_params.keys():
            params_copy['optimizer_params'][k] = optimizer_params[k]

    if 'base_name' in params_copy['model_params'].keys():
        print('BASE NAME', params_copy['model_params']['base_name'])
    if args.drop == 0:
        print('LR to be used: ', params_copy['learning_rate_params']['learning_rate'])
    else:
        print('LR to be used: ', params_copy['learning_rate_params']['next_lr'])
    return params_copy


def convrnn_model(inputs, input_args=[], units=128, *args, **kwargs):
    input_dict = {'images': inputs}
    
    default_args = "--tnn_json=10Lv9p2_imnet224_rrg128ctx_batchnorm_allones --const_pres=True --ff_bn=True --rnn_bn_all_convs=True --gate_tau_bn_gamma_init=0.1 --unroll_tf=True --times=7"
    input_args.extend(default_args.split())
    all_params = train_median_wfb(edges_arr=edges_5, input_args=input_args)

    model_params = all_params['model_params']
    model_params['out_layers'] = 'conv10'
    model_params['base_name'] = os.path.join(
            CONVRNN_REPO_PATH, 
            model_params['base_name'])
    func = model_params.pop('func')
    model_params.update(kwargs)
    output, _ = func(input_dict, **model_params)
    output = tf.reduce_mean(output, [1, 2], keep_dims=False, name='global_pool')
    output = tf.layers.dense(
            inputs=output, units=units, 
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            )
    return output


def convrnn_imagenet_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,7,8,9'
    input_args = ['--gpu', '1,2,3,4,5,7,8,9']
    all_params = train_median_wfb(
            edges_arr=edges_5, 
            input_args=input_args)

    old_model_params = all_params.pop('model_params')
    def _temp_model_func(inputs, *args, **kwargs):
        output = convrnn_model(
                inputs['images'], input_args=input_args, 
                units=1000,
                *args, **kwargs)
        return output, {}
    new_model_params = {
            'func': _temp_model_func,
            'devices': old_model_params['devices'],
            'num_gpus': old_model_params['num_gpus'],
            }
    all_params['model_params'] = new_model_params
    all_params['save_params'] = {
            'host': 'localhost',
            'port': 27009,
            'dbname': 'convrnn',
            'collname': 'control',
            'exp_id': 'cate',
            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': 1000,
            'save_valid_freq': 10009,
            'save_filters_freq': 100090,
            'cache_filters_freq': 100090,
            }
    all_params['load_params'] = {
            'host': 'localhost',
            'port': 27009,
            'dbname': 'convrnn',
            'collname': 'control',
            'exp_id': 'cate',
            'do_restore': True,
            }
    all_params['validation_params'] = {}
    print(all_params.keys())
    base.train_from_params(**all_params)


if __name__ == '__main__':
    #inputs = tf.zeros(shape=[128, 224, 224, 3], dtype=tf.float32)
    #print(convrnn_model(inputs))
    convrnn_imagenet_test()
