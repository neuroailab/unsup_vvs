import os, sys
import numpy as np

import tensorflow as tf
from tfutils import base

import json
import copy
import utils
import pdb

import optimizer
import combinet_builder
from tpu_data_provider import TPUCombineWorld as CombineWorldDataset
import tpu_dp_params
from cmd_parser import get_parser, load_setting
import utilities.data_path_utils as data_path_utils
import models.tpu_loss_utils as tpu_loss_utils 
from models.loss_utils import LossBuilder
import models.config_parser as model_cfg_parser
from models.mean_teacher_utils import rampup_rampdown_lr
from models.model_builder import ModelBuilder


class ParamSetter(object):
    """
    Process the command-line arguments, set up the parameters for training
    """

    def __init__(self, args):
        if args.n_gpus==None:
            args.n_gpus = len(args.gpu.split(','))

        self.args = args
        self._set_configs()

    def __load_dataset_configs(self, config_name):
        if not os.path.exists(config_name):
            config_name = os.path.join('dataset_configs', config_name)
        cfg_dataset = json.load(open(config_name))
        return cfg_dataset

    def _set_configs(self):
        if self.args.topndconfig is None:
            self.args.topndconfig = self.args.valdconfig
        if self.args.featdconfig is None:
            self.args.featdconfig = self.args.valdconfig
        if self.args.modeldconfig is None:
            self.args.modeldconfig = self.args.valdconfig

    def get_save_params(self):
        args = self.args

        exp_id = args.expId
        self.exp_id = exp_id
        dbname = args.dbname
        self.dbname = dbname
        collname = args.collname
        self.collname = collname

        if args.fre_cache_filter is None:
            args.fre_cache_filter = args.fre_filter
        if args.tpu_task:
            collname = 'combinet-tpu'

        save_to_gfs = [
            'fea_image_scenenet', 'fea_normal_scenenet', 'fea_depth_scenenet', 
            'out_normal_scenenet', 'out_depth_scenenet',
            'fea_image_pbrnet', 'fea_normal_pbrnet', 'fea_depth_pbrnet', 
            'out_normal_pbrnet', 'out_depth_pbrnet',
            'fea_image_scannet', 'fea_depth_scannet', 'out_depth_scannet',
            'fea_instance_pbrnet', 'out_instance_pbrnet',
            'fea_instance_scenenet', 'out_instance_scenenet',
            'fea_instance_coco', 'out_instance_coco', 'fea_image_coco',
            'fea_image_nyuv2', 'fea_depth_nyuv2', 'out_depth_nyuv2',
            'fea_image_imagenet', 'out_normal_imagenet', 'out_depth_imagenet',
            'fea_image_place', 'out_normal_place', 'out_depth_place',
            'out_image_imagenet',
        ]

        if args.tpu_task:
            save_params = {
                    'host': 'localhost',
                    'port': args.nport,
                    'dbname': dbname,
                    'collname': collname,
                    'exp_id': exp_id,
                    'do_save': True,
                    'save_valid_freq': args.fre_valid,
                    'save_filters_freq': args.fre_filter,
                    'cache_dir': os.path.join(args.cacheDirPrefix, exp_id),
                    'checkpoint_max': args.checkpoint_max,
                    }
        else:
            #cache_dir = os.path.join(
            #        args.cacheDirPrefix, '.tfutils', 'localhost:27017',
            #        'normalnet-test', 'normalnet', exp_id)
            cache_dir = os.path.join(
                    args.cacheDirPrefix, '.tfutils', 
                    'localhost:%i' % args.nport,
                    dbname, collname, exp_id)
            save_params = {
                    'host': 'localhost',
                    'port': args.nport,
                    'dbname': dbname,
                    'collname': collname,
                    'exp_id': exp_id,
                    'do_save': True,
                    'save_initial_filters': True,
                    'save_metrics_freq': args.fre_metric,
                    'save_valid_freq': args.fre_valid,
                    'save_filters_freq': args.fre_filter,
                    'cache_filters_freq': args.fre_cache_filter,
                    'cache_dir': cache_dir,
                    'save_to_gfs': save_to_gfs,
                    }
        return save_params

    def _get_special_load_dict_for_mt(self, ckpt_file):
        args = self.args
        reader = tf.train.NewCheckpointReader(ckpt_file)
        var_shapes = reader.get_variable_to_shape_map()
        load_param_dict = {}
        for each_key in var_shapes:
            if args.instance_task and 'category' in each_key:
                new_key = each_key.replace('category', 'memory')
                load_param_dict[each_key] = 'primary/%s' % new_key
            else:
                load_param_dict[each_key] = 'primary/%s' % each_key
        return load_param_dict

    def get_load_params(self):
        args = self.args

        loadport = args.nport
        if args.loadport:
            loadport = args.loadport
        loadexpId = self.exp_id
        if args.loadexpId:
            loadexpId = args.loadexpId
        load_query = None
        if args.loadstep:
            load_query = {
                    'exp_id': loadexpId, 
                    'saved_filters': True, 
                    'step': args.loadstep}

        if args.tpu_task:
            load_params = {
                    'do_restore': False,
                    'query': None
                    }
        else:
            load_params = {
                    'host': 'localhost',
                    'port': loadport,
                    'dbname': args.load_dbname or self.dbname,
                    'collname': args.load_collname or self.collname,
                    'exp_id': loadexpId,
                    'do_restore': True,
                    'query': load_query,
                    'from_ckpt': args.ckpt_file,
                    'restore_global_step': not args.drop_global_step,
                    }

            if args.ckpt_file is not None and args.mt_ckpt_load_dict==1:
                load_params['load_param_dict'] \
                        = self._get_special_load_dict_for_mt(args.ckpt_file)
            if args.load_param_dict is not None:
                load_params['load_param_dict'] \
                        = json.loads(args.load_param_dict)
        return load_params

    def _get_instance_params(self):
        args = self.args
        instance_params = {
                'instance_lbl_pkl': args.inst_lbl_pkl,
                'inst_cate_sep': args.inst_cate_sep,
                'instance_task': args.instance_task,
                'instance_k': args.instance_k,
                'instance_t': args.instance_t,
                'instance_m': args.instance_m,
                'instance_data_len': args.instance_data_len,
                }
        return instance_params

    def _add_gpu_to_model_params(self, model_params):
        args = self.args
        if args.tpu_task:
            tpu_model_params = {
                    'tpu_name':args.tpu_name,
                    'gcp_project':args.gcp_project,
                    'tpu_zone':args.tpu_zone,
                    'tpu_task':args.tpu_task,
                    'num_shards':args.tpu_num_shards,
                    }
            model_params.update(tpu_model_params)
        model_params['num_gpus'] = args.n_gpus
        model_params['devices'] \
                = ['/gpu:%i' % (i + args.gpu_offset) \
                   for i in range(args.n_gpus)]
        return model_params

    def _get_simclr_model_params(self):
        import models.simclr_builder as simclr_builder
        model_params = {
                'func': simclr_builder.build,
                }
        model_params = self._add_gpu_to_model_params(model_params)
        return model_params

    def _get_simclr_mid_model_params(self):
        import models.simclr_builder as simclr_builder
        model_params = {
                'func': simclr_builder.build_with_mid,
                }
        model_params = self._add_gpu_to_model_params(model_params)
        return model_params

    def _get_prednet_model_params(self):
        import models.prednet_builder as prednet_builder
        model_params = {
                'func': prednet_builder.build,
                'model_type': self.args.namefunc.split(':')[1],
                'which_layer': self.args.namefunc.split(':')[2],
                }
        model_params = self._add_gpu_to_model_params(model_params)
        return model_params

    def _get_l3_prednet_model_params(self, which_layer='layer3'):
        import models.prednet_builder as prednet_builder
        if which_layer == 'layer3':
            model_params = {
                    'func': prednet_builder.l3_build,
                    }
        elif which_layer == 'layer2':
            model_params = {
                    'func': prednet_builder.l3_build,
                    'wanted_layers': ['A_2', 'Ahat_2', 'E_2', 'R_2'],
                    'pool_ksize': 4, 'pool_stride': 4,
                    }
        else:
            raise NotImplementedError()
        if self.args.tpu_task:
            model_params['tpu'] = True
        model_params = self._add_gpu_to_model_params(model_params)
        return model_params

    def get_cleaned_model_params(self):
        args = self.args
        self.cfg_dataset = self.__load_dataset_configs(args.dataconfig)

        val_target = utils.get_val_target(self.cfg_dataset)
        self.val_target = val_target

        cfg_dataset_model = self.cfg_dataset
        if not args.modeldconfig is None:
            cfg_dataset_model = self.__load_dataset_configs(args.modeldconfig)

        self.model_builder = ModelBuilder(args, cfg_dataset_model)

        if args.namefunc == 'simclr_func':
            return self._get_simclr_model_params()
        if args.namefunc == 'simclr_func_mid':
            return self._get_simclr_mid_model_params()
        if args.namefunc.startswith('prednet'):
            return self._get_prednet_model_params()
        if args.namefunc == 'l3_prednet':
            return self._get_l3_prednet_model_params()
        if args.namefunc == 'l3_prednet_layer2':
            return self._get_l3_prednet_model_params('layer2')

        #model_params = {'func': model_builder.build}
        model_params = {
                'func': lambda *args, **kwargs: self.model_builder.build(
                    *args, **kwargs)}
        model_params = self._add_gpu_to_model_params(model_params)
        return model_params

    def get_model_params(self):
        args = self.args

        func_net = getattr(combinet_builder, args.namefunc)
        cfg_initial = model_cfg_parser.get_network_cfg(args)
        self.cfg_dataset = self.__load_dataset_configs(args.dataconfig)

        val_target = utils.get_val_target(self.cfg_dataset)
        self.val_target = val_target

        model_params = {
                'func': func_net,
                'seed': args.seed,
                'cfg_initial': cfg_initial,
                'cfg_dataset': self.cfg_dataset,
                'init_stddev': args.init_stddev,
                'ignorebname': args.ignorebname,
                'ignorebname_new': args.ignorebname_new,
                'add_batchname': args.add_batchname,
                'weight_decay': args.weight_decay,
                'global_weight_decay': args.global_weight_decay,
                'init_type': args.init_type,
                'cache_filter': args.cache_filter,
                'fix_pretrain': args.fix_pretrain,
                'extra_feat': args.extra_feat,
                'color_norm': args.color_norm,
                'corr_bypassadd': args.corr_bypassadd,
                'mean_teacher': args.mean_teacher==1,
                'ema_decay': args.ema_decay,
                'ema_zerodb': args.ema_zerodb==1,
                'sm_fix': args.sm_fix,
                'sm_de_fix': args.sm_de_fix,
                'sm_depth_fix': args.sm_depth_fix,
                'sm_resnetv2': args.sm_resnetv2,
                'sm_resnetv2_1': args.sm_resnetv2_1,
                'sm_bn_trainable': args.sm_bn_trainable==1,
                'sm_bn_fix': args.sm_bn_fix,
                'tpu_flag': args.tpu_flag,
                'combine_tpu_flag': args.combine_tpu_flag,
                'tpu_depth': args.tpu_depth,
                'tpu_tl_imagenet':args.tpu_tl_imagenet,
                'down_sample': args.col_down,
                'col_knn': args.col_knn==1,
                'color_dp_tl': args.color_dp_tl==1,
                'rp_dp_tl': args.rp_dp_tl==1,
                'depth_imn_tl': args.depth_imn_tl==1,
                'use_lasso': args.use_lasso,
                'combine_col_rp': args.combine_col_rp,
                'train_anyway':args.train_anyway,
                }

        instance_params = self._get_instance_params()
        model_params.update(instance_params)

        if args.tpu_task:
            tpu_model_params = {
                    'tpu_name':args.tpu_name,
                    'gcp_project':args.gcp_project,
                    'tpu_zone':args.tpu_zone,
                    'tpu_task':args.tpu_task,
                    'num_shards':8,
                    }
            model_params.update(tpu_model_params)

        if args.namefunc in [
                'combine_normal_tfutils_new', 
                'combine_tfutils_general'] and args.no_prep==1:
            model_params['no_prep'] = 1

        if not args.modeldconfig is None:
            cfg_dataset_model = self.__load_dataset_configs(args.modeldconfig)
            model_params['cfg_dataset'] = cfg_dataset_model

        model_params['num_gpus'] = args.n_gpus
        model_params['devices'] \
                = ['/gpu:%i' % (i + args.gpu_offset) \
                   for i in range(args.n_gpus)]
        return model_params

    def _get_tpu_dp_params(self):
        args = self.args

        self.tpu_gnrl_dp_tasks = [
                'mean_teacher', 
                'instance_task',
                'imagenet',
                'cpc',
                'multi_imagenet',
                ]
        if args.tpu_task in self.tpu_gnrl_dp_tasks:
            tpu_data_path = tpu_dp_params.get_TPU_data_path()
            tpu_data_prms = {
                    'data_path': tpu_data_path,
                    }
            tpu_data_prms.update(self.data_gnrl_params)
            self.tpu_data_prms = tpu_data_prms
            data_provider_func = CombineWorldDataset(
                    group='train',
                    **tpu_data_prms).input_fn
        else:
            data_provider_func \
                    = tpu_dp_params.get_deprecated_tpu_train_dp_params(args)

        self.train_data_param = {
                'func': data_provider_func,
                'batch_size': args.batchsize}
        return self.train_data_param
        
    def _get_train_data_param(self):
        args = self.args
        which_imagenet = utils.which_imagenet_map(args.whichimagenet)

        self.data_gnrl_params = {
                'which_imagenet': which_imagenet,
                'cfg_dataset': self.cfg_dataset,
                'file_shuffle': args.no_shuffle==0,
                'resnet_prep': args.resnet_prep,
                'resnet_prep_size': args.resnet_prep_size,
                }
        if args.instance_task or args.imgnt_w_idx:
            self.data_gnrl_params['imgnt_w_idx'] = True

        if args.tpu_task:
            train_data_param = self._get_tpu_dp_params()
        else:
            data_path = data_path_utils.get_data_path(
                localimagenet=args.localimagenet,
                overall_local=args.overall_local)
            self.dataset_data_prms = {
                    'data_path': data_path,
                    }
            self.dataset_data_prms.update(self.data_gnrl_params)
            data_provider_func = CombineWorldDataset(
                    group='train',
                    **self.dataset_data_prms).get_input_dict
            train_data_param = {
                    'func': data_provider_func,
                    'batch_size': args.batchsize}
        return train_data_param

    def __need_rep_and_updates(self):
        cfg_dataset = self.cfg_dataset
        for key, value in cfg_dataset.items():
            if 'instance' in key and value == 1:
                return True
        if self.args.with_rep == 1:
            return True
        if self.args.instance_task:
            return True
        return False

    def get_train_params(self):
        args = self.args

        NUM_BATCHES_PER_EPOCH = 5000
        train_params = {}
        train_data_param = self._get_train_data_param()
        if args.tpu_task:
            train_params = {
                    'data_params': train_data_param,
                    'num_steps': args.train_num_steps \
                                 or 1900000 * NUM_BATCHES_PER_EPOCH,
                    }
        else:
            train_params = {
                    'validate_first': False,
                    'data_params': train_data_param,
                    'thres_loss': float('Inf'),
                    'num_steps': args.train_num_steps \
                                 or 1900000 * NUM_BATCHES_PER_EPOCH,
                    }

        if self.__need_rep_and_updates():
            train_params['targets'] = {
                    'func': self.loss_builder.get_rep_losses_and_updates,
                    'devices': self.model_params['devices'],
                    }

        if args.valid_first==1:
            if args.tpu_task is None:
                train_params['validate_first'] = True
            else:
                train_params['tpu_validate_first'] = True
        train_params['train_loop'] = {
                'func': lambda *args, **kwargs: self.model_builder.train_loop(
                    *args, **kwargs)}
        return train_params

    def get_loss_params(self):
        args = self.args
        
        if args.tpu_task:
            return tpu_loss_utils.get_tpu_loss_params(args)

        loss_func_kwargs = {
                'cfg_dataset': self.cfg_dataset,
                'depth_norm': args.depth_norm,
                'depthloss': args.depthloss,
                'normalloss': args.normalloss,
                'extra_feat': args.extra_feat,
                'sm_half_size': args.sm_half_size,
                'mean_teacher': args.mean_teacher==1,
                'res_coef': args.res_coef,
                'cons_ramp_len': args.cons_ramp_len,
                'cons_max_value': args.cons_max_value,
                'instance_task': args.instance_task,
                'instance_k': args.instance_k,
                'instance_t': args.instance_t,
                'instance_data_len': args.instance_data_len,
                'inst_cate_sep': args.inst_cate_sep,
                }
        self.loss_func_kwargs = loss_func_kwargs
        self.loss_builder = LossBuilder(**loss_func_kwargs)
        loss_func = self.loss_builder.get_loss

        loss_params = {
                'targets': self.val_target,
                'loss_per_case_func': loss_func,
                'loss_per_case_func_params': {},
                'loss_func_kwargs': loss_func_kwargs,
                'inputs_as_dict': True,
                }
        if args.gpu_task=='colorization':
            loss_params['loss_per_case_func'] = utils.gpu_col_loss
        elif args.gpu_task=='color_tl':
            loss_params['loss_per_case_func'] = utils.gpu_col_tl_loss
        return loss_params

    def get_learning_rate_params(self):
        args = self.args
        learning_rate_params = {
                'func': lambda global_step, learning_rate: \
                        tf.constant(learning_rate),
                'learning_rate': args.init_lr,
                }

        if args.lr_boundaries is not None:
            boundaries = args.lr_boundaries.split(',')
            boundaries = [int(each_boundary) for each_boundary in boundaries]

            all_lrs = [
                    args.init_lr * (0.1 ** drop_level) \
                    for drop_level in range(len(boundaries) + 1)]

            learning_rate_params = {
                    'func':lambda global_step, boundaries, values:\
                            tf.train.piecewise_constant(
                                x=global_step,
                                boundaries=boundaries, values=values),
                    'values': all_lrs,
                    'boundaries': boundaries,
                    }

        if args.tpu_task=='mean_teacher' or args.mt_ramp_down == 1:
            learning_rate_params = {
                    'func': rampup_rampdown_lr,
                    'init_lr': args.init_lr,
                    'target_lr': args.batchsize / 16 * args.target_lr,
                    'nb_per_epoch': 1280000 / args.batchsize,
                    'enable_ramp_down': args.mt_ramp_down == 1,
                    'ramp_down_epoch': args.mt_ramp_down_epoch,
                    }
        return learning_rate_params

    def get_optimizer_params(self):
        args = self.args

        optimizer_class = tf.train.MomentumOptimizer
        clip_flag = args.withclip==1
        optimizer_params = {
                'func': optimizer.ClipOptimizerSelf,
                'optimizer_class': optimizer_class,
                'clip': clip_flag,
                'momentum': .9,
                }

        if args.whichopt==1:
            optimizer_params = {
                    'func': optimizer.ClipOptimizerSelf,
                    'optimizer_class': tf.train.AdamOptimizer,
                    'clip': clip_flag,
                    'epsilon': args.adameps,
                    'beta1': args.adambeta1,
                    'beta2': args.adambeta2,
                    }
        if args.whichopt==2:
            optimizer_params = {
                'func': optimizer.ClipOptimizerSelf,
                'optimizer_class': tf.train.AdagradOptimizer,
                'clip': clip_flag,
            }

        if args.whichopt==3:
            optimizer_params = {
                    'func': optimizer.ClipOptimizerSelf,
                    'optimizer_class': optimizer_class,
                    'clip': clip_flag,
                    'momentum': .9,
                    'use_nesterov': True,
                    }
        if args.whichopt==4:
            optimizer_params = {
                    'func': optimizer.ClipOptimizerSelf,
                    'optimizer_class': tf.train.AdadeltaOptimizer,
                    'clip': clip_flag,
                    }
        if args.whichopt==5:
            optimizer_params = {
                    'func': optimizer.ClipOptimizerSelf,
                    'optimizer_class': tf.train.RMSPropOptimizer,
                    'clip': clip_flag,
                    }
        if args.whichopt==6:
            optimizer_params = {
                    'func': optimizer.ClipOptimizerSelf,
                    'optimizer_class': tf.train.GradientDescentOptimizer,
                    'clip': clip_flag,
                    }
        if args.whichopt==7:
            optimizer_params = {
                    'func': optimizer.ClipOptimizerSelf,
                    'optimizer_class': optimizer_class,
                    'clip': clip_flag,
                    'momentum': .999,
                    }

        if args.trainable_scope:
            optimizer_params['trainable_scope'] = [args.trainable_scope]
        return optimizer_params

    def _get_val_topn_data_params(self):
        args = self.args

        val_batch_size = args.batchsize
        if not args.minibatch is None:
            val_batch_size = args.minibatch
        if not args.valbatchsize is None:
            val_batch_size = args.valbatchsize
        self.val_batch_size = val_batch_size

        val_input_func = CombineWorldDataset(
                group='val',
                **self.dataset_data_prms).get_input_dict
        self.topn_val_data_param = {
                'func': val_input_func,
                'batch_size': self.val_batch_size}
        return self.topn_val_data_param

    def _get_val_tpu_topn_data_params(self):
        args = self.args

        val_batch_size = args.batchsize
        if not args.minibatch is None:
            val_batch_size = args.minibatch
        if not args.valbatchsize is None:
            val_batch_size = args.valbatchsize
        self.val_batch_size = val_batch_size

        if args.tpu_task in self.tpu_gnrl_dp_tasks:
            val_input_fn = CombineWorldDataset(
                    group='val',
                    **self.tpu_data_prms).input_fn
        else:
            val_input_fn \
                    = tpu_dp_params.get_deprecated_val_tpu_topn_dp_params(args)

        self.topn_val_data_param = {
                'func': val_input_fn,
                'batch_size': self.val_batch_size,
                }
        return self.topn_val_data_param

    def _get_tpu_validation_targets(self):
        args = self.args
        loss_rep_targets = {'func': tpu_loss_utils.metric_fn}

        if args.tpu_task=='rp' \
                or args.tpu_task=='rp_pbr' \
                or args.tpu_task=='rp_only_pbr':
            loss_rep_targets = {'func': tpu_loss_utils.rp_metric_fn}
        elif args.tpu_task=='colorization' \
                or args.tpu_task=='color_ps' \
                or args.tpu_task=='color_pbr':
            loss_rep_targets = {'func': tpu_loss_utils.col_metric_fn}
        elif args.tpu_task=='depth' or args.tpu_task=='depth_pbr':
            loss_rep_targets = {'func': tpu_loss_utils.depth_metric_fn}
        elif args.tpu_task=='combine_depth_imn':
            loss_rep_targets = {
                    'func': tpu_loss_utils.combine_depth_imn_metric_fn}
        elif args.tpu_task=='combine_rp_imn' \
                or args.tpu_task=='combine_rp_col' \
                or args.tpu_task=='combine_rdc' \
                or args.tpu_task=='combine_rdc_imn' \
                or args.tpu_task=='combine_rp_col_ps' \
                or args.tpu_task=='combine_rci':
            loss_rep_targets = {'func': tpu_loss_utils.combine_rp_imn_metric_fn}
        elif args.tpu_task=='mean_teacher':
            loss_rep_targets = {
                    'func':tpu_loss_utils.tpu_mean_teacher_metric_fn}
        elif args.tpu_task=='instance_task':
            loss_rep_targets = {'func':tpu_loss_utils.tpu_instance_metric_fn}
        elif args.tpu_task=='cpc':
            def _cpc_metric_fn(labels, logits, **kwargs):
                return {'val_loss': tf.metrics.mean(logits)}
            loss_rep_targets = {'func': _cpc_metric_fn}
        else:
            raise NotImplementedError
        return loss_rep_targets

    def _get_gpu_validation_targets(self):
        args = self.args

        cfg_dataset = self.cfg_dataset
        if not args.topndconfig is None:
            cfg_dataset_topn = self.__load_dataset_configs(args.topndconfig)
            cfg_dataset = cfg_dataset_topn

        loss_rep_targets = {
                'func': self.loss_builder.get_val_metrics,
                'cfg_dataset': cfg_dataset}
        if args.gpu_task=='colorization':
            loss_rep_targets = {'func':utils.gpu_col_val}
        elif args.gpu_task=='color_tl':
            loss_rep_targets = {'func':utils.gpu_col_tl_val}
        return loss_rep_targets

    def _get_topn_val_param(self):
        args = self.args
        val_step_num = 1000
        if args.valinum > -1:
            val_step_num = args.valinum
        topn_val_param = {}
        if args.tpu_task:
            topn_val_param = {
                    'data_params': self._get_val_tpu_topn_data_params(),
                    'targets': self._get_tpu_validation_targets(),
                    'num_steps': val_step_num
                    }
        else:
            topn_val_param = {
                'data_params': self._get_val_topn_data_params(),
                'targets': self._get_gpu_validation_targets(),
                'num_steps': val_step_num,
                'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
                'online_agg_func': utils.online_agg
                }    

        return topn_val_param

    def _get_feat_val_targets(self):
        feat_val_targets = {
                'func': self.loss_builder.get_feat_targets,
                'num_to_save': 5
                }
        return feat_val_targets

    def _get_feat_val_param(self):
        args = self.args
        feat_val_param = {
            'data_params': self._get_val_topn_data_params(),
            'targets': self._get_feat_val_targets(),
            'num_steps': 10,
            'agg_func': utils.mean_losses_keep_rest,
            }    
        return feat_val_param

    def _get_pca_val_targets(self):
        pca_val_targets = {
                'func': self.loss_builder.get_pca_targets,
                }
        return pca_val_targets

    def _get_pca_val_param(self):
        args = self.args
        pca_val_param = {
                'data_params': self._get_val_topn_data_params(),
                'targets': self._get_pca_val_targets(),
                'num_steps': args.valinum,
                'agg_func': lambda res: self.model_builder.compute_pca(res),
                }
        return pca_val_param

    def get_validation_params(self):
        args = self.args
        validation_params = {}
        
        if args.validation_skip==1:
            return validation_params

        topn_val_param = self._get_topn_val_param()
        validation_params = {
                'topn': topn_val_param,
                }
        if args.with_feat == 1:
            feat_val_param = self._get_feat_val_param()
            validation_params['feat'] = feat_val_param
        if args.do_pca == 1:
            pca_val_param = self._get_pca_val_param()
            validation_params = {'pca': pca_val_param}
        return validation_params

    def build_all_params(self):
        self.save_params = self.get_save_params()
        self.load_params = self.get_load_params()
        self.model_params = self.get_cleaned_model_params()
        self.loss_params = self.get_loss_params()
        self.train_params = self.get_train_params()
        self.validation_params = self.get_validation_params()
        self.learning_rate_params = self.get_learning_rate_params()
        self.optimizer_params = self.get_optimizer_params()
        
        params = {
                'save_params': self.save_params,
                'load_params': self.load_params,
                'model_params': self.model_params,
                'train_params': self.train_params,
                'loss_params': self.loss_params,
                'learning_rate_params': self.learning_rate_params,
                'optimizer_params': self.optimizer_params,
                'log_device_placement': False,
                'validation_params': self.validation_params,
                'skip_check': True,
                }
        return params
    
    def build_test_params(self):
        self.save_params = self.get_save_params()
        self.load_params = self.get_load_params()
        self.model_params = self.get_cleaned_model_params()
        _ = self.get_loss_params()
        _ = self.get_train_params()
        self.validation_params = self.get_validation_params()

        params = {
            'save_params': self.save_params,
            'load_params': self.load_params,
            'model_params': self.model_params,
            'validation_params': self.validation_params,
            'skip_check': True,
        }
        return params


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args = load_setting(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    param_setter = ParamSetter(args)
    params = param_setter.build_all_params()
    base.train_from_params(**params)
