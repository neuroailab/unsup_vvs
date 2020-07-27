from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np
import tensorflow as tf
import cPickle

import json
import copy
import argparse
import time
import functools
import inspect

from tfutils import base, optimizer
import tfutils.defaults

from model import preprocessing as prep
from model import instance_model
from model.memory_bank import MemoryBank
from model.dataset_utils import dataset_func

from utils import DATA_LEN_IMAGENET_FULL, tuple_get_one
import config


PreprocessFunc = config.named_choices({
    'resnet': prep.resnet_train,
    'center_crop': prep.resnet_validate,
    'resnet_crop_only': prep.resnet_crop_only,
    'resnet_noflip': prep.resnet_noflip,
    'resnet_nocrop': prep.resnet_nocrop,
    'resnet_bigcrop': prep.resnet_bigcrop,
    'resnet_4way_rot': prep.resnet_4way_rot,
    'resnet_rot': prep.resnet_rot,
})


def get_config():
    cfg = config.Config()
    cfg.add('exp_id', type=str, required=True,
            help='Name of experiment ID')
    cfg.add('batch_size', type=int, default=128,
            help='Training batch size')
    cfg.add('test_batch_size', type=int, default=64,
            help='Testing batch size')
    cfg.add('init_lr', type=float, default=0.03,
            help='Initial learning rate')
    cfg.add('target_lr', type=float, default=None,
            help='Target leraning rate for ramping up')
    cfg.add('ramp_up_epoch', type=int, default=1,
            help='Number of epoch for ramping up')
    cfg.add('gpu', type=str, required=True,
            help='Value for CUDA_VISIBLE_DEVICES')
    cfg.add('image_dir', type=str, required=True,
            help='Directory containing dataset')
    cfg.add('q_cap', type=int, default=102400,
            help='Shuffle queue capacity of tfr data')
    cfg.add('data_len', type=int, default=DATA_LEN_IMAGENET_FULL,
            help='Total number of images in the input dataset')
    cfg.add('preprocess_func', type=PreprocessFunc, default='resnet',
            help='Specifies which kind of data augmentation to'
                 ' use (defaults to resnet augmentation)')

    # Training parameters
    cfg.add('weight_decay', type=float, default=1e-4,
            help='Weight decay')
    cfg.add('instance_t', type=float, default=0.07,
            help='Temperature in softmax.')
    cfg.add('instance_k', type=int, default=4096,
            help='Closes neighbors to sample.')
    cfg.add('lr_boundaries', type=str, default=None,
            help='Learning rate boundaries for 10x drops')

    cfg.add('use_clusters', default=None, type=str,
            help='Path to npy file containing cluster labels.')
    cfg.add('num_cluster_samples', default=1, type=int,
            help='Number of samples from the cluster to use if using clusters.')
    cfg.add('add_topn_dot', default=None, type=int,
            help='None (default) means not adding, number means how many')
    cfg.add('nn_list_path', type=str,
            help='Path to .npz file storing nearest neighbors '
                 'which will be used instead of recomputing.')
    cfg.add('add_thres_dot', default=None, type=float,
            help='None (default) means not adding, number means the threshold')
    cfg.add('maximize_log_cluster_prob', type=bool,
            help='Flag for using the negative log cluster probability as loss.')
    cfg.add('resnet_size', type=int, default=18,
            help='ResNet size')
    cfg.add('cluster_method', type=str, default='NN',
            help='Cluster method, KM and NN supported for now')
    cfg.add('clstr_path', type=str, default='NN',
            help='Known labels, for SEMI method')
    cfg.add('clstr_update_fre', type=int, default=None,
            help='Update frequency for cluster')
    cfg.add('kmeans_k', type=str, default='10000',
            help='K for Kmeans')
    cfg.add('model_type', type=str, default='resnet',
            help='Model type, resnet or alexnet')
    cfg.add('multi_type', type=str, default='separate',
            help='How to aggregate different kmeans')

    # Saving parameters
    cfg.add('port', type=int, required=True,
            help='Port number for mongodb')
    cfg.add('host', type=str, default='localhost',
            help='Host for mongodb')
    cfg.add('db_name', type=str, required=True,
            help='Name of database')
    cfg.add('col_name', type=str, required=True,
            help='Name of collection')
    cfg.add('cache_dir', type=str, required=True,
            help='Prefix of cache directory for tfutils')
    cfg.add('fre_valid', type=int, default=10009,
            help='Frequency of validation')
    cfg.add('fre_metric', type=int, default=1000,
            help='Frequency of saving metrics')
    cfg.add('fre_filter', type=int, default=10009,
            help='Frequency of saving filters')
    cfg.add('fre_cache_filter', type=int,
            help='Frequency of caching filters')

    # Loading parameters
    cfg.add('load_exp', type=str, default=None,
            help='The experiment to load from, in the format '
                 '[dbname]/[collname]/[exp_id]')
    cfg.add('load_port', type=int,
            help='Port number of mongodb for loading (defaults to saving port')
    cfg.add('load_step', type=int,
            help='Step number for loading')
    cfg.add('resume', type=bool,
            help='Flag for loading from last step of this exp_id, will override'
            ' all other loading options.')

    return cfg


def tfutils_func_params(func, to_record, **kwargs):
    '''
    Helper for creating parameters describing a function to be passed
    to tfutils.
    '''
    for k in to_record:
        if k not in kwargs:
            raise Exception("Cannot record parameter %r which "
                            "does not appear in kwargs." % k)

    params, partial_kwargs = {}, {}
    for k, v in kwargs.items():
        if k in to_record:
            # These parameters will be recorded by tfutils.
            params[k] = v
        else:
            partial_kwargs[k] = v

    # Similar to functools.partial, but we do it this way to retain
    # some module info (which tfutils assumes exists).
    @functools.wraps(func)
    def partial_func(**kw):
        kw.update(partial_kwargs)
        return func(**kw)
    assert 'func' not in params
    params['func'] = partial_func
    return params


def loss_func(output, *args, **kwargs):
    # Get pure loss
    loss_pure, loss_model, loss_noise = output['losses']
    # Filter the weights in agg_func (reg_loss_in_faster) later
    return loss_pure


def reg_loss_in_faster(loss, which_device, weight_decay):
    from tfutils.multi_gpu.easy_variable_mgr import COPY_NAME_SCOPE
    curr_scope_name = '%s%i' % (COPY_NAME_SCOPE, which_device)
    # Add weight decay to the loss.
    def exclude_batch_norm_and_other_device(name):
        return 'batch_normalization' not in name and curr_scope_name in name
    l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32))
                for v in tf.trainable_variables()
                if exclude_batch_norm_and_other_device(v.name)])
    loss_all = tf.add(loss, l2_loss)
    return loss_all


def rep_loss_func(
        inputs,
        output,
        **kwargs
        ):
    data_indx = output['data_indx']
    new_data_memory = output['new_data_memory']
    loss_pure, loss_model, loss_noise = output['losses']

    memory_bank_list = output['memory_bank']
    all_labels_list = output['all_labels']
    if isinstance(memory_bank_list, tf.Variable):
        memory_bank_list = [memory_bank_list]
        all_labels_list = [all_labels_list]

    nns = output['nns']
    new_nns = output['new_nns']

    devices = ['/gpu:%i' % idx for idx in range(len(memory_bank_list))]
    update_ops = []
    for device, memory_bank, all_labels, nn \
            in zip(devices, memory_bank_list, all_labels_list, nns):
        with tf.device(device):
            mb_update_op = tf.scatter_update(
                    memory_bank, data_indx, new_data_memory)
            update_ops.append(mb_update_op)
            lb_update_op = tf.scatter_update(
                    all_labels, data_indx,
                    inputs['label'])
            update_ops.append(lb_update_op)
            update_ops.append(tf.scatter_update(
                nn, data_indx, new_nns))

    with tf.control_dependencies(update_ops):
        # Force the updates to happen before the next batch.
        loss_model = tf.identity(loss_model)
        loss_noise = tf.identity(loss_noise)

    return {
            'loss_model': loss_model,
            'loss_noise': loss_noise,
            }


def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res


def valid_perf_func(
        inputs,
        output,
        ):
    curr_dist, all_labels = output
    all_labels = tuple_get_one(all_labels)
    _, top_indices = tf.nn.top_k(curr_dist, k=1)
    curr_pred = tf.gather(all_labels, tf.squeeze(top_indices, axis=1))
    imagenet_top1 = tf.reduce_mean(
            tf.cast(
                tf.equal(curr_pred, inputs['label']),
                tf.float32))
    return {'top1': imagenet_top1}


def get_model_func_params(args):
    model_params = {
        "instance_data_len": args.data_len,
        "instance_t": args.instance_t,
        "instance_k": args.instance_k,
        "add_topn_dot": args.add_topn_dot,
        "add_thres_dot": args.add_thres_dot,
        "nn_list_path": args.nn_list_path,
        "use_clusters": args.use_clusters,
        "num_cluster_samples": args.num_cluster_samples,
        "maximize_log_cluster_prob": args.maximize_log_cluster_prob,
        "resnet_size": args.resnet_size,
        "cluster_method": args.cluster_method,
        "kmeans_k": args.kmeans_k,
        "model_type": args.model_type,
        "clstr_path": args.clstr_path,
        "multi_type": args.multi_type,
    }
    return model_params


def get_lr_from_boundary_and_ramp_up(
        global_step, boundaries, 
        init_lr, target_lr, ramp_up_epoch,
        NUM_BATCHES_PER_EPOCH):
    curr_epoch  = tf.div(
            tf.cast(global_step, tf.float32), 
            tf.cast(NUM_BATCHES_PER_EPOCH, tf.float32))
    curr_phase = (tf.minimum(curr_epoch/float(ramp_up_epoch), 1))
    curr_lr = init_lr + (target_lr-init_lr) * curr_phase

    if boundaries is not None:
        boundaries = boundaries.split(',')
        boundaries = [int(each_boundary) for each_boundary in boundaries]

        all_lrs = [
                curr_lr * (0.1 ** drop_level) \
                for drop_level in range(len(boundaries) + 1)]

        curr_lr = tf.train.piecewise_constant(
                x=global_step,
                boundaries=boundaries, values=all_lrs)
    return curr_lr



def get_params_from_arg(args):
    '''
    This function gets parameters needed for tfutils.train_from_params()
    '''
    multi_gpu = len(args.gpu.split(','))
    data_len = args.data_len
    val_len = 50000
    NUM_BATCHES_PER_EPOCH = data_len // args.batch_size

    # save_params: defining where to save the models
    args.fre_cache_filter = args.fre_cache_filter or args.fre_filter
    cache_dir = os.path.join(
            args.cache_dir, '.tfutils', 'localhost:%i' % args.port,
            args.db_name, args.col_name, args.exp_id)
    save_params = {
            'host': 'localhost',
            'port': args.port,
            'dbname': args.db_name,
            'collname': args.col_name,
            'exp_id': args.exp_id,
            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': args.fre_metric,
            'save_valid_freq': args.fre_valid,
            'save_filters_freq': args.fre_filter,
            'cache_filters_freq': args.fre_cache_filter,
            'cache_dir': cache_dir,
            }

    # load_params: defining where to load, if needed
    load_port = args.load_port or args.port
    load_dbname = args.db_name
    load_collname = args.col_name
    load_exp_id = args.exp_id
    load_query = None

    if not args.resume:
        if args.load_exp is not None:
            load_dbname, load_collname, load_exp_id = args.load_exp.split('/')
        if args.load_step:
            load_query = {'exp_id': load_exp_id,
                          'saved_filters': True,
                          'step': args.load_step}
            print('Load query', load_query)

    load_params = {
            'host': 'localhost',
            'port': load_port,
            'dbname': load_dbname,
            'collname': load_collname,
            'exp_id': load_exp_id,
            'do_restore': True,
            'query': load_query,
            }


    # XXX: hack to set up training loop properly
    if args.kmeans_k.isdigit():
        args.kmeans_k = [int(args.kmeans_k)]
    else:
        args.kmeans_k = [int(each_k) for each_k in args.kmeans_k.split(',')]
    nn_clusterings = []
    # model_params: a function that will build the model
    model_func_params = get_model_func_params(args)
    def build_output(inputs, train, **kwargs):
        res = instance_model.build_output(inputs, train, **model_func_params)
        if not train:
            return res
        outputs, logged_cfg, clustering = res
        nn_clusterings.append(clustering)
        return outputs, logged_cfg

    def train_loop(sess, train_targets, **params):
        assert len(nn_clusterings) == multi_gpu

        global_step_vars = [v for v in tf.global_variables() if 'global_step' in v.name]
        assert len(global_step_vars) == 1
        global_step = sess.run(global_step_vars[0])

        # TODO: consider making this reclustering frequency a flag
        if global_step % (args.clstr_update_fre or NUM_BATCHES_PER_EPOCH) == 0:
            print("Recomputing clusters...")
            if args.cluster_method == 'NN':
                for clustering in nn_clusterings:
                    clustering.recompute_clusters(sess)
            else:
                new_clust_labels = nn_clusterings[0].recompute_clusters(
                        sess)
                for clustering in nn_clusterings:
                    clustering.apply_clusters(sess, new_clust_labels)

        return tfutils.defaults.train_loop(sess, train_targets, **params)

    model_params = {'func': build_output}
    if multi_gpu > 1:
        model_params['num_gpus'] = multi_gpu
        model_params['devices'] = ['/gpu:%i' % idx for idx in range(multi_gpu)]

    # train_params: parameters about training data
    train_data_param = tfutils_func_params(
        dataset_func, ['image_dir', 'is_train', 'q_cap', 'batch_size'],
        image_dir=args.image_dir,
        process_img_func=args.preprocess_func,
        is_train=True,
        q_cap=args.q_cap,
        batch_size=args.batch_size
    )
    train_params = {
            'validate_first': False,
            'data_params': train_data_param,
            'queue_params': None,
            'thres_loss': float('Inf'),
            'num_steps': 2000 * NUM_BATCHES_PER_EPOCH,
            'train_loop': {'func': train_loop},
            }

    ## Add other loss reports (loss_model, loss_noise)
    train_params['targets'] = {
            'func': rep_loss_func,
            }

    # loss_params: parameters to build the loss
    loss_params = {
        'pred_targets': [],
        'agg_func': reg_loss_in_faster,
        'agg_func_kwargs': {'weight_decay': args.weight_decay},
        'loss_func': loss_func,
    }
    if args.model_type == 'convrnn':
        loss_params.pop('agg_func')
        loss_params.pop('agg_func_kwargs')

    # learning_rate_params: build the learning rate
    # For now, just stay the same
    learning_rate_params = {
            'func': get_lr_from_boundary_and_ramp_up,
            'init_lr': args.init_lr,
            'target_lr': args.target_lr or args.init_lr,
            'NUM_BATCHES_PER_EPOCH': NUM_BATCHES_PER_EPOCH,
            'boundaries': args.lr_boundaries,
            'ramp_up_epoch': args.ramp_up_epoch,
            }

    # optimizer_params: use tfutils optimizer,
    # as mini batch is implemented there
    optimizer_params = {
            'optimizer': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.MomentumOptimizer,
            'clip': False,
            'momentum': .9,
            }
    if args.model_type == 'convrnn':
        optimizer_params['clip'] = True
        optimizer_params['use_nesterov'] = True

    # validation_params: control the validation
    topn_val_data_param = tfutils_func_params(
        dataset_func, ['image_dir', 'is_train', 'q_cap', 'batch_size'],
        image_dir=args.image_dir,
        process_img_func=prep.resnet_validate,
        is_train=False,
        q_cap=args.test_batch_size,
        batch_size=args.test_batch_size
    )
    val_step_num = int(val_len/args.test_batch_size)
    topn_val_param = {
        'data_params': topn_val_data_param,
        'queue_params': None,
        'targets': {'func': valid_perf_func},
        'num_steps': val_step_num,
        'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
        'online_agg_func': online_agg,
        }
    validation_params = {
            'topn': topn_val_param,
            }

    # Put all parameters together
    params = {
            'save_params': save_params,
            'load_params': load_params,
            'model_params': model_params,
            'train_params': train_params,
            'loss_params': loss_params,
            'learning_rate_params': learning_rate_params,
            'optimizer_params': optimizer_params,
            'log_device_placement': False,
            'validation_params': validation_params,
            'skip_check': True,
            }
    return params


def main():
    # Parse arguments
    cfg = get_config()
    args = cfg.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Get params needed, start training
    params = get_params_from_arg(args)
    base.train_from_params(**params)

if __name__ == "__main__":
    main()
