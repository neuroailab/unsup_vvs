from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import pdb

from tfutils import base, optimizer

import json
import copy
import argparse
import h5py
import time

from nf_utils import *
import neural_data_builder
from nf_cmd_parser import get_parser, load_setting
import get_dc_data_params as dc_dp

sys.path.append('../combine_pred/')
import train_combinet
import combinet_builder
import utils as cb_utils
from new_neural_data_provider import NeuralNewDataTF


def get_params_from_arg(args):
    if args.objectome_zip or args.h5py_data_loader:
        assert args.gen_features==1, \
                "Please generate features"

    # Expand it_nodes and v4_nodes, see expand_nodes defined in nf_utils.py
    if args.it_nodes is not None:
        args.it_nodes = expand_nodes(args.it_nodes)
        args.it_nodes = expand_nodes(args.it_nodes)
    if args.v4_nodes is not None:
        args.v4_nodes = expand_nodes(args.v4_nodes)
        args.v4_nodes = expand_nodes(args.v4_nodes)

    cache_dir = os.path.join(
            args.cacheDirPrefix, '.tfutils', 'localhost:27017', 
            'neuralfit-test', 'neuralfit', args.expId)

    if args.ave_or_10ms==0:
        args.f10ms_time = None

    if args.n_gpus == None:
        args.n_gpus = len(args.gpu.split(','))

    # Save params
    save_to_gfs = []
    save_params = {
            'host': 'localhost',
            'port': args.nport,
            'dbname': args.dbname,
            'collname': args.colname,
            'exp_id': args.expId,

            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': args.fre_metric,
            'save_valid_freq': args.fre_valid,
            'save_filters_freq': args.fre_filter,
            'cache_filters_freq': args.fre_filter,
            'cache_dir': cache_dir,
            'save_to_gfs': save_to_gfs,
        }

    # Load params, by default, load from combinet model
    loadport = args.nport
    if not args.loadport is None:
        loadport = args.loadport
    loadexpId = args.expId
    if not args.loadexpId is None:
        loadexpId = args.loadexpId
    load_query = None
    if not args.loadstep is None:
        load_query = {
                'exp_id': loadexpId, 
                'saved_filters': True, 
                'step': args.loadstep}
    print("********load query**********", load_query)
    load_params = {
            'host':'localhost',
            'port':args.loadport,
            'dbname':args.loaddbname,
            'collname':args.loadcolname,
            'exp_id':loadexpId,
            'do_restore':True,
            'query':load_query,
            'from_ckpt': args.ckpt_file,
        }
    if args.deepcluster:
        load_params['do_restore'] = False

    # model params
    cfg_initial = cb_utils.get_network_cfg(
            args, 
            configdir=args.configdir)
    func_net = getattr(neural_data_builder, args.namefunc)
    model_params = {
            'func':func_net,
            'seed':args.seed,
            'cfg_initial':cfg_initial,
            'no_prep':args.no_prep,
            'rp_sub_mean': args.rp_sub_mean,
            'div_std': args.div_std,
            'color_prep': args.color_prep,
            'combine_col_rp': args.combine_col_rp,
            'input_mode': args.input_mode,
            'v4_nodes':args.v4_nodes,
            'it_nodes':args.it_nodes,
            'cache_filter':args.cache_filter,
            'init_type':args.init_type,
            'weight_decay': args.weight_decay,
            'weight_decay_type': args.weight_decay_type,
            'random_sample':args.random_sample,
            'random_sample_seed':args.random_sample_seed,
            'random_gather':args.train_or_test==1,
            'in_conv_form':args.in_conv_form==1,
            'ignorebname_new':args.ignorebname_new,
            'batch_name':args.batch_name,
            'f10ms_time':args.f10ms_time,
            'partnet_train':args.partnet_train==1,
            'gen_features':args.gen_features==1,
            'sm_resnetv2':args.sm_resnetv2,
            'tpu_flag': args.tpu_flag,
            'mean_teacher':args.mean_teacher==1,
            'ema_decay':args.ema_decay,
            'ema_zerodb':args.ema_zerodb==1,
            'fixweights':True, 
            'sm_bn_trainable':False,
            'inst_model':args.inst_model,
            'inst_res_size':args.inst_res_size,
            'spatial_select':args.spatial_select,
            'convrnn_model':args.convrnn_model,
        }
    model_params['num_gpus'] = args.n_gpus
    model_params['devices'] = [\
            '/gpu:%i' % (i + args.gpu_offset) \
            for i in range(args.n_gpus)]
    if args.deepcluster:
        model_params['use_precompute'] = True

    if args.dataset_type == 'v1v2':
        model_params['v4_out_shape'] = 102
        model_params['it_out_shape'] = 103
    elif args.dataset_type == 'v1_cadena':
        model_params['v4_out_shape'] = 166
        model_params['it_out_shape'] = 0

    # train params
    BATCH_SIZE = args.batchsize

    data_param_base = {}
    if not (args.objectome_zip or args.h5py_data_loader):
        if args.ave_or_10ms==0:
            DATA_PATH = get_data_path(
                    image_prefix=args.image_prefix,
                    neuron_resp_prefix=args.neuron_resp_prefix,
                    )
            data_path = {
                    'images': DATA_PATH['%s/images' % args.which_split],
                    'V4_ave': DATA_PATH['%s/V4' % args.which_split],
                    'IT_ave': DATA_PATH['%s/IT' % args.which_split],
                    }
            shape_dict = {
                    'images': (256, 256, 3),
                    'V4_ave': (128,),
                    'IT_ave': (168,),
                    }

            if args.dataset_type == 'v1v2':
                split_dir = os.path.join(args.v1v2_folder, args.which_split)
                data_path = {
                        'images': os.path.join(split_dir, 'images'),
                        'V1_ave': os.path.join(split_dir, 'V1_ave'),
                        'V2_ave': os.path.join(split_dir, 'V2_ave'),
                        }
                shape_dict = {
                        'images': (320, 320, 3),
                        'V1_ave': (102,),
                        'V2_ave': (103,),
                        }
                data_param_base['buffer_size'] = 270
            elif args.dataset_type == 'v1_cadena':
                split_dir = os.path.join(args.v1v2_folder, args.which_split)
                data_path = {
                        'images': os.path.join(split_dir, 'images'),
                        'V1_ave': os.path.join(split_dir, 'V1_ave'),
                        }
                shape_dict = {
                        'images': (80, 80, 3),
                        'V1_ave': (166,),
                        }
                data_param_base['buffer_size'] = 2560
            elif args.dataset_type == 'hvm_var6':
                tfr_dir_path = '/mnt/fs4/chengxuz/v4it_temp_results/var6_tfrecords/'
                data_path = {
                        'images': os.path.join(
                            tfr_dir_path, args.which_split, 'images'),
                        'V4_ave': os.path.join(
                            tfr_dir_path, args.which_split, 'V4_ave'),
                        'IT_ave': os.path.join(
                            tfr_dir_path, args.which_split, 'IT_ave'),
                        }
                data_param_base['buffer_size'] = 1024

            if args.deepcluster == 'vgg16':
                data_path, shape_dict = dc_dp.get_vgg16_params(
                        data_path, shape_dict)
            elif args.deepcluster == 'res18':
                data_path, shape_dict = dc_dp.get_res18_params(
                        data_path, shape_dict)
            elif args.deepcluster in ['res18_deseq', 'res18_deseq_v1']:
                data_path, shape_dict = dc_dp.get_res18_deseq_params(
                        data_path, shape_dict, args)
            elif args.deepcluster in ['res50_deseq', 'res50_deseq_v1']:
                data_path, shape_dict = dc_dp.get_res50_deseq_params(
                        data_path, shape_dict, args)
            elif args.deepcluster is not None \
                    and args.deepcluster.split(':')[0] \
                        in ['cmc_res50', 'cmc_res50_v1']:
                data_path, shape_dict = dc_dp.get_cmc_res50_params(
                        data_path, shape_dict, args)
            elif args.deepcluster in ['cmc_res18', 'cmc_res18_v1']:
                data_path, shape_dict = dc_dp.get_cmc_res18_params(
                        data_path, shape_dict, args)
            elif args.deepcluster in ['la_cmc_res18', 'la_cmc_res18_v1']:
                data_path, shape_dict = dc_dp.get_la_cmc_res18_params(
                        data_path, shape_dict, args)
            elif args.deepcluster in ['la_cmc_res18v1', 'la_cmc_res18v1_v1']:
                data_path, shape_dict = dc_dp.get_la_cmc_res18v1_params(
                        data_path, shape_dict, args)
            elif args.deepcluster in ['pt_official_res18', 
                                      'pt_official_res18_v1', 
                                      'pt_official_res18_var6']:
                data_path, shape_dict = dc_dp.get_pt_official_res18_params(
                        data_path, shape_dict, args)

            def _wrapper_data_func(batch_size, **kwargs):
                data_provider = NeuralNewDataTF(**kwargs)
                return data_provider.input_fn(batch_size)

            data_param_base.update({
                    'func': _wrapper_data_func,
                    'data_path': data_path,
                    'shape_dict': shape_dict,
                    'batch_size': BATCH_SIZE,
                    'img_out_size': args.img_out_size,
                    'img_crop_size': args.img_crop_size,
                    })

            train_data_param = {'group': 'train'}
            train_data_param.update(data_param_base)

        train_params = {
                'validate_first': False,
                'data_params': train_data_param,
                'thres_loss': float('Inf'),
                'num_steps': args.train_num_steps,  # number of steps to train
                }

    # loss params
    if args.ave_or_10ms==0:
        val_target = ['V4_ave', 'IT_ave']
        if args.dataset_type == 'v1v2':
            val_target = ['V1_ave', 'V2_ave']
        elif args.dataset_type == 'v1_cadena':
            val_target = ['V1_ave']
        loss_func_kwargs = {
                'v4_nodes':args.v4_nodes,
                'it_nodes':args.it_nodes,
                }
        loss_func = loss_withcfg
    else:
        val_target = ['labels']
        loss_func_kwargs = {
                'v4_nodes':args.v4_nodes,
                'it_nodes':args.it_nodes,
                'f10ms_time':args.f10ms_time,
                }
        loss_func = loss_withcfg_10ms

    loss_params = {
            'targets': val_target,
            'loss_per_case_func': loss_func,
            'loss_per_case_func_params': {},
            'loss_func_kwargs': loss_func_kwargs,
        }

    # learning rate params
    learning_rate_params = {
            'func': tf.train.exponential_decay,
            'learning_rate': args.init_lr,
            #'decay_rate': .95,
            'decay_rate': 1,
            'decay_steps': 900,  # exponential decay each epoch
            'staircase': True
        }

    # optimizer_params
    optimizer_class = tf.train.MomentumOptimizer
    clip_flag = args.withclip==1

    optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': optimizer_class,
            'clip': clip_flag,
            'momentum': .9
        }

    if args.whichopt==1:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdamOptimizer,
            'clip': clip_flag,
            'epsilon': args.adameps,
            'beta1': args.adambeta1,
            'beta2': args.adambeta2,
        }

    if args.whichopt==2:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdagradOptimizer,
            'clip': clip_flag,
        }

    if args.whichopt==3:
        optimizer_params = {
                'func': optimizer.ClipOptimizer,
                'optimizer_class': optimizer_class,
                'clip': clip_flag,
                'momentum': .9,
                'use_nesterov': True
            }

    if args.whichopt==4:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdadeltaOptimizer,
            'clip': clip_flag,
        }

    if args.whichopt==5:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.RMSPropOptimizer,
            'clip': clip_flag,
        }

    # validation params
    if not (args.objectome_zip or args.h5py_data_loader):
        val_data_param = {'group': 'val'}
        val_data_param.update(data_param_base)
    elif args.objectome_zip:
        import data_objectome
        which_data_provider = data_objectome.dataset_func
        DATA_PATH = args.image_prefix
        val_data_param = {
                'func': which_data_provider,
                'data_path': DATA_PATH,
                'batch_size': BATCH_SIZE,
                'dataset_type': args.obj_dataset_type,
                'cadena_im_size': args.cadena_im_size,
            }
    else:
        import h5py_ventral_neural_data_reader as h5py_loader
        # TODO: make filter func changable according to args
        def _filter_func(idx, hf):
            return hf['image_meta']['variation_level'][idx] in ['V6']
        val_data_param = {
                'func': h5py_loader.dataset_func,
                'data_path': '/mnt/fs4/chengxuz/v4it_temp_results/ventral_neural_data.hdf5',
                'batch_size': BATCH_SIZE,
                'filter_func': _filter_func,
            }

    val_step_num = int(256*5/BATCH_SIZE)
    if args.dataset_type == 'v1v2':
        val_step_num = int(90/BATCH_SIZE)
    elif args.dataset_type == 'v1_cadena':
        val_step_num = int(625/BATCH_SIZE)
    elif args.dataset_type == 'hvm_var6':
        val_step_num = int(256*2/BATCH_SIZE)

    if args.train_or_test==1:
        val_data_param['group'] = 'all'
        val_data_param['shuffle'] = False
        val_step_num = int(256*5*4/BATCH_SIZE)
    if args.gen_features==1:
        if not (args.objectome_zip or args.h5py_data_loader):
            # Only generate features for training examples?
            if args.gen_features_onval==0:
                val_data_param['group'] = 'train'
                val_step_num = int(256*5*3/BATCH_SIZE)
            val_data_param['shuffle'] = False

    if args.ave_or_10ms==0:
        loss_rep_targets = {
                'func':loss_rep_withcfg,
                'target':['V4_ave', 'IT_ave'],
                'v4_nodes':args.v4_nodes,
                'it_nodes':args.it_nodes,
                }
    else:
        loss_rep_targets = {
                'func':loss_rep_withcfg_10ms,
                'target':['labels'],
                'v4_nodes':args.v4_nodes,
                'it_nodes':args.it_nodes,
                'f10ms_time':args.f10ms_time,
                }
    if args.train_or_test==1 or args.gen_features==1:
        loss_rep_targets = {
                'func':loss_gather,
                }
    l2loss_val_param = {
                'data_params': val_data_param,
                'targets': loss_rep_targets,
                'num_steps': val_step_num,
                #'agg_func': lambda x: {k:np.mean(v) for k,v in x.items()},
                'agg_func': agg_func,
                #'online_agg_func': online_agg,
                'online_agg_func': online_agg_corr,
            }
    if args.train_or_test==1:
        l2loss_val_param['agg_func'] = agg_func_pls
        l2loss_val_param['online_agg_func'] = online_agg_append
    if args.gen_features==1:
        l2loss_val_param['online_agg_func'] = online_agg_append

    validation_params = {
            'l2loss_val':l2loss_val_param
            }
    if args.val_on_train==1:
        val_on_train_step_num = int(256*5*3/BATCH_SIZE)
        if args.dataset_type == 'v1v2':
            val_on_train_step_num = int(360/BATCH_SIZE)
        elif args.dataset_type == 'v1_cadena':
            val_on_train_step_num = int(5616/BATCH_SIZE)
        elif args.dataset_type == 'hvm_var6':
            val_on_train_step_num = int(256*8/BATCH_SIZE)

        l2_loss_train_param = {
                'data_params': train_data_param,
                'targets': loss_rep_targets,
                'num_steps': val_on_train_step_num,
                'agg_func': agg_func,
                'online_agg_func': online_agg_corr,
                }
        validation_params['l2loss_train'] = l2_loss_train_param

    # Final params
    if args.train_or_test==0 and args.gen_features==0:
        params = {
            'save_params': save_params,
            'load_params': load_params,
            'model_params': model_params,
            'train_params': train_params,
            'loss_params': loss_params,
            'learning_rate_params': learning_rate_params,
            'optimizer_params': optimizer_params,
            'validation_params': validation_params,
            'skip_check': True,
        }
        if not args.minibatch is None:
            params['train_params']['minibatch_size'] = args.minibatch

    else:
        params = {
            'save_params': save_params,
            'load_params': load_params,
            'model_params': model_params,
            'validation_params': validation_params,
            'skip_check': True,
        }
        if args.gen_features==1:
            params['dont_run'] = True

    return params


def main():
    parser = get_parser()
    args = parser.parse_args()
    args = load_setting(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = get_params_from_arg(args)

    if not params is None:
        if args.gen_features==0:
            if args.train_or_test==0:
                base.train_from_params(**params)
            else:
                base.test_from_params(**params)
        else:
            test_args = base.test_from_params(**params)

            dbinterface = test_args['dbinterface'][0]
            sess = dbinterface.sess
            validation_targets = test_args['validation_targets'][0]

            if not (args.objectome_zip or args.h5py_data_loader):
                if args.gen_features_onval==0:
                    over_num = 256*5*3
                else:
                    over_num = 256*5
                over_num = over_num*2
            elif args.objectome_zip:
                import data_objectome
                if args.obj_dataset_type == 'objectome':
                    over_num = data_objectome.OBJECTOME_DATA_LEN
                elif args.obj_dataset_type == 'v1_cadena':
                    over_num = data_objectome.V1_CADENA_DATA_LEN
            else:
                # Number of images for V6
                over_num = 2560

            val_step_num = int(np.ceil(over_num * 1.0/args.batchsize))

            now_num = 0
            hdf5_dir_name = os.path.dirname(args.gen_hdf5path)
            if not os.path.isdir(hdf5_dir_name):
                os.system('mkdir -p ' + hdf5_dir_name)
            fout = h5py.File(args.gen_hdf5path, 'w')
            for indx_tmp in range(val_step_num):
                start_time = time.time()
                res = sess.run(validation_targets['l2loss_val']['targets'])
                end_num = min(now_num + args.batchsize, over_num)

                for key_tmp in res:
                    now_data = res[key_tmp]
                    if key_tmp not in fout:
                        new_shape = list(now_data.shape)
                        new_shape[0] = over_num
                        dataset_tmp = fout.create_dataset(
                                key_tmp, new_shape, dtype='f')
                    else:
                        dataset_tmp = fout[key_tmp]
                    dataset_tmp[now_num : end_num] \
                            = now_data[:(end_num - now_num)]
                    
                now_num = end_num
                end_time = time.time()
                print('Batch %i takes time %f' \
                        % (indx_tmp, end_time - start_time))

            fout.close()
            sess.close()


if __name__ == '__main__':
    main()
