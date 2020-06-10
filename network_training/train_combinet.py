from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf

from tfutils import base, optimizer

import json
import copy
import utils
import pdb

from tpu_data_provider import TPUCombineWorld
from cmd_parser import get_parser, load_setting

# Separate data providers for TPU data
from tpu_old_dps.full_imagenet_input import ImageNetInput
from tpu_old_dps.rp_imagenet_input import RP_ImageNetInput
from tpu_old_dps.rp_pbrscenenet_input import PBRSceneNetDepthMltInput
from tpu_old_dps.col_imagenet_input import Col_ImageNetInput
from tpu_old_dps.col_pbrscenenet_input import Col_PBRSceneNetInput
from tpu_old_dps.col_pbr_input import Col_PBRNetInput
from tpu_old_dps.depth_pbrscenenet_input import PBRSceneNetDepthInput
from tpu_old_dps.depth_pbr_input import PBRNetDepthInput
from tpu_old_dps.rp_pbr_input import PBRNetDepthMltInput
from tpu_old_dps.rp_ps_zip_input import PBRSceneNetZipInput
from tpu_old_dps.depth_pbr_zip_input import PBRNetZipDepthInput
from tpu_old_dps.depth_ps_zip_input import PBRSceneNetZipDepthInput
from tpu_old_dps.col_tl_imagenet_input import Col_Tl_Input
from tpu_old_dps.combine_depth_imn_input import DepthImagenetInput
from tpu_old_dps.combine_rp_imagenet_input import Combine_RP_ImageNet_Input 
from tpu_old_dps.combine_rp_col_input import Combine_RP_Color_Input
from tpu_old_dps.combine_rci_input import Combine_RCI_Input
from tpu_old_dps.combine_rp_col_ps_input import Combine_RP_Color_PS_Input
from tpu_old_dps.combine_rp_col_input_new import Combine_RP_Color_Input_New
from tpu_old_dps.combine_rdc_input import Combine_RDC_Input
from tpu_old_dps.combine_rdc_imn_input import Combine_RDC_ImageNet_Input

sys.path.append('../normal_pred/')
import normal_encoder_asymmetric_with_bypass
import combinet_builder

BATCH_SIZE = 32


def get_params_from_arg(args):
    if args.topndconfig is None:
        args.topndconfig = args.valdconfig
    if args.featdconfig is None:
        args.featdconfig = args.valdconfig
    if args.modeldconfig is None:
        args.modeldconfig = args.valdconfig

    if args.n_gpus==None:
        args.n_gpus = len(args.gpu.split(','))

    cfg_initial = utils.get_network_cfg(args)

    dataconfig = args.dataconfig
    if not os.path.exists(dataconfig):
        dataconfig = os.path.join('dataset_configs', dataconfig)
    cfg_dataset = utils.postprocess_config(json.load(open(dataconfig)))

    exp_id = args.expId
    cache_dir = os.path.join(args.cacheDirPrefix, '.tfutils', 'localhost:27017', \
            'normalnet-test', 'normalnet', exp_id)

    BATCH_SIZE = normal_encoder_asymmetric_with_bypass.getBatchSize(cfg_initial)
    queue_capa = normal_encoder_asymmetric_with_bypass.getQueueCap(cfg_initial)
    if args.batchsize != None:
        BATCH_SIZE = args.batchsize

    if args.queuecap != None:
        queue_capa = args.queuecap

    func_net = getattr(combinet_builder, args.namefunc)

    if args.depthnormal==1:
        assert args.depth_norm==1, "Depth norm needs to be 1!"

    DATA_PATH = utils.get_data_path(
        localimagenet=args.localimagenet,
        overall_local=args.overall_local)
    
    batchsize_data_param = 1
    if args.tpu_task:
        batchsize_data_param = BATCH_SIZE
    which_imagenet = utils.which_imagenet_map(args.whichimagenet)
    data_param_base = {
            'data_path': DATA_PATH,
            'batch_size': batchsize_data_param,
            'cfg_dataset': cfg_dataset,
            'depthnormal': args.depthnormal,
            'depthnormal_div': args.depthnormal_div,
            'withflip': args.withflip,
            'whichimagenet': which_imagenet,
            'whichcoco': args.whichcoco,
            'crop_time': args.crop_time,
            'crop_rate': args.crop_rate,
            'with_color_noise': args.with_color_noise,
            'prob_gray':args.prob_gray,
            'size_minval':args.size_minval,
            'which_place': args.which_place,
            'size_vary_prep': args.size_vary_prep,
            'fix_asp_ratio': args.fix_asp_ratio,
            'img_size_vary_prep': args.img_size_vary_prep,
            'sm_full_size': args.sm_full_size,  # sm_add
            'crop_size': args.crop_size,
            'col_size': args.col_size,
            'val_on_train':args.val_on_train,
            }

    train_data_param_base = {
            'group': 'train',
            'shuffle_seed': args.shuffle_seed,
            'no_shuffle': args.no_shuffle,
            'replace_folder': args.replace_folder_train,
            }
    train_data_param_base.update(data_param_base)

    train_queue_params = {
            'queue_type': 'random',
            'batch_size': BATCH_SIZE,
            'seed': 0,
            'capacity': queue_capa,
            'min_after_dequeue': BATCH_SIZE,
            }
    Combine_world = None
    if args.use_dataset_inter or args.tpu_task:
        train_queue_params = None
    else:
        from data_provider import Combine_world as _tmp_func
        Combine_world = _tmp_func

    tpu_data_prms = {}
    dataset_data_prms = {}
    tpu_data_path = utils.get_TPU_data_path()

    # Set general parameters for TPUCombineWorld
    tpu_data_gnrl_params = {
            'which_imagenet': which_imagenet,
            'cfg_dataset': cfg_dataset,
            'file_shuffle': args.no_shuffle==0,
            'resnet_prep': args.resnet_prep,
            'resnet_prep_size': args.resnet_prep_size,
            }
    ## For instance task
    if args.instance_task:
        tpu_data_gnrl_params['imgnt_w_idx'] = True

    # Tasks in this list will use general tpu data providers
    tpu_gnrl_dp_tasks = [
            'mean_teacher', 
            'instance_task',
            'imagenet',
            ]
    # tpu_task is not None means we are using tpu for the tasks
    if args.tpu_task:
        if args.tpu_task=='imagenet_rp':
            data_provider_func = ImageNetInput(
                    True, args.sm_loaddir, 
                    std=False).input_fn
        if args.tpu_task=='rp':
            data_provider_func = RP_ImageNetInput(
                    True, args.sm_loaddir, 
                    g_noise=args.g_noise, std=(args.rp_std==1), 
                    sub_mean=(args.rp_sub_mean==1), 
                    grayscale=(args.rp_grayscale==1)).input_fn
        if args.tpu_task=='rp_pbr':
            data_provider_func = PBRSceneNetZipInput(
                    True, args.sm_loaddir, 
                    args.sm_loaddir2, g_noise=args.g_noise, 
                    std=(args.rp_std==1)).input_fn
            if args.rp_zip==0:
                data_provider_func = PBRSceneNetDepthMltInput(
                        True, args.sm_loaddir, 
                        args.sm_loaddir2, g_noise=args.g_noise, 
                        std=(args.rp_std==1)).input_fn
        if args.tpu_task=='rp_only_pbr':
            data_provider_func = PBRNetDepthMltInput(
                    True, args.sm_loaddir, 
                    g_noise=args.g_noise, std=(args.rp_std==1)).input_fn
        if args.tpu_task=='colorization':
            data_provider_func = Col_ImageNetInput(
                    True, args.sm_loaddir, 
                    down_sample=args.col_down, col_knn=args.col_knn==1, 
                    col_tl=(args.col_tl==1), 
                    combine_rp=(args.combine_rp==1)).input_fn
        if args.tpu_task=='color_ps':
             data_provider_func = Col_PBRSceneNetInput(
                     True, args.sm_loaddir, 
                     args.sm_loaddir2, down_sample=args.col_down, 
                     col_knn=args.col_knn==1).input_fn  
        if args.tpu_task=='color_pbr':
             data_provider_func = Col_PBRNetInput(
                     True, args.sm_loaddir, 
                     down_sample=args.col_down, 
                     col_knn=args.col_knn==1).input_fn 
        if args.tpu_task=='color_tl':
             data_provider_func = Col_Tl_Input(
                     True, args.sm_loaddir, 
                     down_sample=args.col_down, 
                     col_knn=args.col_knn==1, 
                     combine_rp=(args.combine_rp==1)).input_fn 
        if args.tpu_task=='depth':
            data_provider_func = PBRSceneNetZipDepthInput(
                    True, args.sm_loaddir, args.sm_loaddir2, 
                    ab_depth=(args.ab_depth==1), down_sample=args.depth_down, 
                    color_dp_tl=(args.color_dp_tl==1), 
                    rp_dp_tl=(args.rp_dp_tl==1), 
                    rpcol_dp_tl=(args.combine_col_rp==1)).input_fn
            if args.depth_zip== 0:
                data_provider_func = PBRSceneNetDepthInput(
                        True, args.sm_loaddir, args.sm_loaddir2).input_fn
        if args.tpu_task=='combine_rp_imn':
            data_provider_func = Combine_RP_ImageNet_Input(
                    True, args.sm_loaddir).input_fn
        if args.tpu_task=='combine_rp_col':
            data_provider_func = Combine_RP_Color_Input(
                    True, args.sm_loaddir, num_grids=1).input_fn
        if args.tpu_task=='combine_rci':
            data_provider_func = Combine_RCI_Input(
                    True, args.sm_loaddir, 
                    num_grids=1).input_fn
        if args.tpu_task=='combine_rp_col_ps':
            data_provider_func = Combine_RP_Color_PS_Input(
                    True, args.sm_loaddir, 
                    args.sm_loaddir2, num_grids=1).input_fn
        if args.tpu_task=='combine_rdc':
            data_provider_func = Combine_RDC_Input(
                    True, args.sm_loaddir, 
                    args.sm_loaddir2).input_fn
        if args.tpu_task=='combine_rdc_imn':
            data_provider_func = Combine_RDC_ImageNet_Input(
                    True, args.sm_loaddir, 
                    args.sm_loaddir2, args.sm_loaddir3).input_fn

        # Tasks using general data providers
        if args.tpu_task in tpu_gnrl_dp_tasks:
            tpu_data_prms = {
                    'data_path': tpu_data_path,
                    }
            tpu_data_prms.update(tpu_data_gnrl_params)
            data_provider_func = TPUCombineWorld(
                    group='train',
                    **tpu_data_prms).input_fn

        # Build the data param
        train_data_param = {
                'func': data_provider_func,
                'batch_size': args.batchsize}

    else: # tpu_task is None
        if not args.use_dataset_inter:
            # Use previous queue interface
            train_data_param = {
                    'func': Combine_world,}
            train_data_param.update(train_data_param_base)
        else:
            # Use new dataset interface
            dataset_data_prms = {
                    'data_path': DATA_PATH,
                    }
            dataset_data_prms.update(tpu_data_gnrl_params)
            data_provider_func = TPUCombineWorld(
                    group='train',
                    **dataset_data_prms).get_input_dict

            train_data_param = {
                    'func': data_provider_func,
                    'batch_size': args.batchsize}

    val_data_param = {
            'func': Combine_world,
            'group': 'val',
            'replace_folder': args.replace_folder_val,}
    val_data_param.update(data_param_base)

    val_batch_size = BATCH_SIZE
    if not args.minibatch is None:
        val_batch_size = args.minibatch
    if not args.valbatchsize is None:
        val_batch_size = args.valbatchsize
    val_queue_params = {
            'queue_type': 'fifo',
            'batch_size': val_batch_size,
            'seed': 0,
            'capacity': val_batch_size * 2,
            'min_after_dequeue': val_batch_size,
            }
    if args.use_dataset_inter:
        val_queue_params = None
         

    val_target = utils.get_val_target(cfg_dataset)
    val_step_num = 1000
    NUM_BATCHES_PER_EPOCH = 5000
    if args.valinum > -1:
        val_step_num = args.valinum

    loss_func = utils.loss_withcfg
    if args.tpu_task=='imagenet' \
            or args.tpu_task=='imagenet_rp' \
            or args.tpu_task=='color_tl':
        loss_func_pre = utils.tpu_imagenet_loss
    if args.tpu_task=='rp':
        loss_func_pre = utils.tpu_rp_imagenet_loss
    if args.tpu_task=='rp_pbr' or args.tpu_task=='rp_only_pbr':
        loss_func_pre = utils.tpu_rp_imagenet_loss
    if args.tpu_task=='colorization' \
            or args.tpu_task=='color_ps' \
            or args.tpu_task=='color_pbr':
        loss_func_pre = utils.tpu_col_loss
    if args.tpu_task=='depth' or args.tpu_task=='depth_pbr':
        loss_func_pre = utils.tpu_depth_loss
    if args.tpu_task=='combine_depth_imn':
        loss_func_pre = utils.combine_depth_imn_loss
    if args.tpu_task=='combine_rp_imn' \
            or args.tpu_task=='combine_rp_col' \
            or args.tpu_task=='combine_rdc' \
            or args.tpu_task=='combine_rdc_imn' \
            or args.tpu_task=='combine_rp_col_ps' \
            or args.tpu_task=='combine_rci':
        loss_func_pre = utils.combine_rp_imn_loss
    if args.tpu_task=='mean_teacher':
        loss_func_pre = utils.tpu_mean_teacher_loss
    if args.tpu_task=='instance_task':
        def _instance_loss(logits, labels, **kwargs):
            loss_pure, _, _ = utils.instance_loss(
                    logits[0], logits[1], 
                    instance_k=args.instance_k,
                    instance_data_len=args.instance_data_len,)
            return loss_all
        loss_func_pre = _instance_loss
    if args.tpu_task:
        def _wrapper_wd(*args, **kwargs):
            loss_pure = loss_func_pre(*args, **kwargs)
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if len(reg_losses)!=0:
                loss_all = tf.add(loss_pure, tf.reduce_sum(reg_losses))
            return loss_all
        loss_func = _wrapper_wd

    loss_func_kwargs = {
            'cfg_dataset': cfg_dataset,
            'depth_norm': args.depth_norm,
            'label_norm': args.label_norm,
            'depthloss': args.depthloss,
            'normalloss': args.normalloss,
            'multtime': args.multtime,
            'extra_feat': args.extra_feat,
            'sm_half_size': args.sm_half_size,
            'mean_teacher': args.mean_teacher==1,
            'res_coef': args.res_coef,
            'cons_ramp_len': args.cons_ramp_len,
            'cons_max_value': args.cons_max_value,
            'instance_task': args.instance_task,
            'instance_k': args.instance_k,
            'instance_data_len': args.instance_data_len,
            'inst_cate_sep': args.inst_cate_sep,
            }
    
    if args.tpu_task:
        loss_params = {
                'agg_func': tf.reduce_mean,
                'loss_per_case_func': loss_func,
                'loss_per_case_func_params': {},
                'loss_func_kwargs': {},
                }
        if args.tpu_task=='mean_teacher':
            loss_params['loss_func_kwargs'] = {
                    'res_coef':args.res_coef,
                    'cons_ramp_len':args.cons_ramp_len,
                    'cons_max_value':args.cons_max_value,
                    'mt_infant_loss':args.mt_infant_loss,
                    }
    else:
        loss_params = {
                'targets': val_target,
                'loss_per_case_func': loss_func,
                'loss_per_case_func_params': {},
                'loss_func_kwargs': loss_func_kwargs,
                }
        if args.gpu_task=='colorization':
            loss_params['loss_per_case_func'] = utils.gpu_col_loss
        elif args.gpu_task=='color_tl':
            loss_params['loss_per_case_func'] = utils.gpu_col_tl_loss

    learning_rate_params = {
            'func': tf.train.exponential_decay,
            'learning_rate': args.init_lr,
            'decay_rate': 1,
            'decay_steps': 6 * NUM_BATCHES_PER_EPOCH,
            'staircase': True
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
                'func': utils.rampup_rampdown_lr,
                'init_lr': args.init_lr,
                'target_lr': BATCH_SIZE/16*args.target_lr,
                'nb_per_epoch': 1280000/BATCH_SIZE,
                'enable_ramp_down': args.mt_ramp_down == 1,
                'ramp_down_epoch': args.mt_ramp_down_epoch,
                }

    model_params = {
            'func': func_net,
            'seed': args.seed,
            'cfg_initial': cfg_initial,
            'cfg_dataset': cfg_dataset,
            'init_stddev': args.init_stddev,
            'ignorebname': args.ignorebname,
            'ignorebname_new': args.ignorebname_new,
            'add_batchname':args.add_batchname,
            'weight_decay': args.weight_decay,
            'global_weight_decay': args.global_weight_decay,
            'init_type': args.init_type,
            'cache_filter': args.cache_filter,
            'fix_pretrain': args.fix_pretrain,
            'extra_feat': args.extra_feat,
            'color_norm': args.color_norm,
            'corr_bypassadd': args.corr_bypassadd,
            'mean_teacher':args.mean_teacher==1,
            'ema_decay':args.ema_decay,
            'ema_zerodb':args.ema_zerodb==1,
            'instance_task':args.instance_task,
            'instance_k':args.instance_k,
            'instance_t':args.instance_t,
            'instance_m':args.instance_m,
            'instance_data_len':args.instance_data_len,
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
            'instance_lbl_pkl':args.inst_lbl_pkl,
            'inst_cate_sep': args.inst_cate_sep,
            }

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
        dataconfig = args.modeldconfig
        if not os.path.exists(dataconfig):
            dataconfig = os.path.join('dataset_configs', dataconfig)
        cfg_dataset_model = utils.postprocess_config(
                json.load(open(dataconfig)))
        model_params['cfg_dataset'] = cfg_dataset_model

    model_params['num_gpus'] = args.n_gpus
    model_params['devices'] = \
            ['/gpu:%i' % (i + args.gpu_offset) for i in xrange(args.n_gpus)]

    dbname = args.dbname
    collname = args.collname

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
    ]

    if args.with_rep==1:
        save_to_gfs.extend(\
                ['loss_normal_scenenet', 'loss_depth_scenenet', 
                    'loss_instance_scenenet', 'loss_depth_scannet',
                    'loss_normal_pbrnet', 'loss_depth_pbrnet', 
                    'loss_instance_pbrnet',
                    'loss_top1_imagenet', 'loss_top5_imagenet',
                    'loss_instance_coco',
                    'loss_top1_place', 'loss_top5_place', 
                    'loss_depth_nyuv2',
                    ])

    if args.fre_cache_filter is None:
        args.fre_cache_filter = args.fre_filter

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

    loadport = args.nport
    if args.loadport:
        loadport = args.loadport
    loadexpId = exp_id
    if args.loadexpId:
        loadexpId = args.loadexpId
    load_query = None
    if args.loadstep:
        load_query = {
                'exp_id': loadexpId, 'saved_filters': True, 
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
                'dbname': args.load_dbname or dbname,
                'collname': args.load_collname or collname,
                'exp_id': loadexpId,
                'do_restore': True,
                'query': load_query,
                'from_ckpt': args.ckpt_file,
                }

        if args.ckpt_file is not None and args.mt_ckpt_load_dict==1:
            reader = tf.train.NewCheckpointReader(args.ckpt_file)
            var_shapes = reader.get_variable_to_shape_map()
            load_param_dict = {}
            for each_key in var_shapes:
                #if not each_key.startswith('encode'):
                #    continue
                if args.instance_task and 'category' in each_key:
                    new_key = each_key.replace('category', 'memory')
                    load_param_dict[each_key] = 'primary/%s' % new_key
                else:
                    load_param_dict[each_key] = 'primary/%s' % each_key

            load_params['load_param_dict'] = load_param_dict

    loss_rep_targets_kwargs = {
            'target': val_target,
            'cfg_dataset': cfg_dataset,
            'depth_norm': args.depth_norm,
            'depthloss': args.depthloss,
            'normalloss': args.normalloss,
            'multtime': args.multtime,
            'extra_feat': args.extra_feat,
            'sm_half_size': args.sm_half_size,
            'mean_teacher':args.mean_teacher==1,
            'instance_task':args.instance_task==1,
            'inst_cate_sep': args.inst_cate_sep,
            }
    topn_val_data_param = copy.deepcopy(val_data_param)

    if not args.topndconfig is None:
        dataconfig = args.topndconfig
        if not os.path.exists(dataconfig):
            dataconfig = os.path.join('dataset_configs', dataconfig)
        cfg_dataset_topn = utils.postprocess_config(
                json.load(open(dataconfig)))
        val_target_topn = utils.get_val_target(cfg_dataset_topn)

        loss_rep_targets_kwargs['cfg_dataset'] = cfg_dataset_topn
        loss_rep_targets_kwargs['target'] = val_target_topn
        topn_val_data_param['cfg_dataset'] = cfg_dataset_topn
    
    if args.tpu_task:
        if args.tpu_task=='imagenet_rp':
            val_input_fn = ImageNetInput(False, args.sm_loaddir, std=False).input_fn

        if args.tpu_task=='rp':
            val_input_fn = RP_ImageNetInput(False, args.sm_loaddir).input_fn

        if args.tpu_task=='rp_pbr':
            val_input_fn =  PBRSceneNetZipInput(
                    False, args.sm_loaddir, args.sm_loaddir2).input_fn
            if args.rp_zip==0:
                val_input_fn = PBRSceneNetDepthMltInput(
                        False, args.sm_loaddir, args.sm_loaddir2).input_fn

        if args.tpu_task=='rp_only_pbr':
            val_input_fn = PBRNetDepthMltInput(False, args.sm_loaddir).input_fn

        if args.tpu_task=='colorization':
            val_input_fn = Col_ImageNetInput(
                    False, args.sm_loaddir, down_sample=args.col_down, 
                    col_knn=(args.col_knn==1), col_tl=(args.col_tl==1)).input_fn

        if args.tpu_task=='color_ps':
            val_input_fn = Col_PBRSceneNetInput(
                    False, args.sm_loaddir, args.sm_loaddir2, 
                    down_sample=args.col_down, col_knn=(args.col_knn==1)).input_fn

        if args.tpu_task=='color_pbr':
            val_input_fn = Col_PBRNetInput(
                    False, args.sm_loaddir, 
                    down_sample=args.col_down, col_knn=(args.col_knn==1)).input_fn

        if args.tpu_task=='color_tl':
            val_input_fn = Col_Tl_Input(
                    False, args.sm_loaddir, 
                    down_sample=args.col_down, col_knn=(args.col_knn==1), 
                    combine_rp=(args.combine_rp==1)).input_fn

        if args.tpu_task=='depth':
            val_input_fn = PBRSceneNetZipDepthInput(
                    False, args.sm_loaddir, args.sm_loaddir2, 
                    ab_depth=(args.ab_depth==1), down_sample=args.depth_down, 
                    color_dp_tl=(args.color_dp_tl==1), rp_dp_tl=(args.rp_dp_tl==1), 
                    rpcol_dp_tl=(args.combine_col_rp==1)).input_fn

            if args.depth_zip==0:
                val_input_fn = PBRSceneNetDepthInput(
                        False, args.sm_loaddir, args.sm_loaddir2).input_fn

        if args.tpu_task=='depth_pbr':
            val_input_fn = PBRNetZipDepthInput(False, args.sm_loaddir).input_fn

            if args.depth_zip==0:
                val_input_fn = PBRNetDepthInput(False, args.sm_loaddir).input_fn

        if args.tpu_task=='combine_depth_imn':
            val_input_fn = DepthImagenetInput(
                    False, args.sm_loaddir, args.sm_loaddir2).input_fn

        if args.tpu_task=='combine_rp_imn':
            val_input_fn = Combine_RP_ImageNet_Input(False, args.sm_loaddir).input_fn

        if args.tpu_task=='combine_rp_col':
            val_input_fn = Combine_RP_Color_Input(
                    False, args.sm_loaddir, num_grids=1).input_fn
        
        if args.tpu_task=='combine_rci':
            val_input_fn = Combine_RCI_Input(False, args.sm_loaddir, num_grids=1).input_fn
        
        if args.tpu_task=='combine_rp_col_ps':
            val_input_fn = Combine_RP_Color_PS_Input(
                    False, args.sm_loaddir, args.sm_loaddir2, num_grids=1).input_fn

        if args.tpu_task=='combine_rdc':
            val_input_fn = Combine_RDC_Input(
                    False, args.sm_loaddir, args.sm_loaddir2).input_fn
        if args.tpu_task=='combine_rdc_imn':
            val_input_fn = Combine_RDC_ImageNet_Input(
                    False, args.sm_loaddir, args.sm_loaddir2, args.sm_loaddir3).input_fn
        
        # For mean teacher task
        if args.tpu_task in tpu_gnrl_dp_tasks:
            val_input_fn = TPUCombineWorld(
                    group='val',
                    **tpu_data_prms).input_fn

        topn_val_data_param = {
                'func': val_input_fn,
                'batch_size': val_batch_size,
                }

    if args.use_dataset_inter:
        val_input_func = TPUCombineWorld(
                group='val',
                **dataset_data_prms).get_input_dict
        topn_val_data_param = {
                'func': val_input_func,
                'batch_size': val_batch_size}

    if args.tpu_task:
        loss_rep_targets = {'func': utils.metric_fn}

        if args.tpu_task=='rp' \
                or args.tpu_task=='rp_pbr' \
                or args.tpu_task=='rp_only_pbr':
            loss_rep_targets = {'func': utils.rp_metric_fn}
        elif args.tpu_task=='colorization' \
                or args.tpu_task=='color_ps' \
                or args.tpu_task=='color_pbr':
            loss_rep_targets = {'func': utils.col_metric_fn}
        elif args.tpu_task=='depth' or args.tpu_task=='depth_pbr':
            loss_rep_targets = {'func': utils.depth_metric_fn}
        elif args.tpu_task=='combine_depth_imn':
            loss_rep_targets = {'func': utils.combine_depth_imn_metric_fn}
        elif args.tpu_task=='combine_rp_imn' \
                or args.tpu_task=='combine_rp_col' \
                or args.tpu_task=='combine_rdc' \
                or args.tpu_task=='combine_rdc_imn' \
                or args.tpu_task=='combine_rp_col_ps' \
                or args.tpu_task=='combine_rci':
            loss_rep_targets = {'func': utils.combine_rp_imn_metric_fn}
        elif args.tpu_task=='mean_teacher':
            loss_rep_targets = {'func':utils.tpu_mean_teacher_metric_fn}
        elif  args.tpu_task=='instance_task':
            loss_rep_targets = {'func':utils.tpu_instance_metric_fn}
    else:
        loss_rep_targets = {'func': utils.rep_loss_withcfg}
        loss_rep_targets.update(loss_rep_targets_kwargs)
        if args.gpu_task=='colorization':
            loss_rep_targets = {'func':utils.gpu_col_val}
        elif args.gpu_task=='color_tl':
            loss_rep_targets = {'func':utils.gpu_col_tl_val}
    
    if args.tpu_task:
        train_params = {
                'data_params': train_data_param,
                'num_steps': 1900000 * NUM_BATCHES_PER_EPOCH,
                }
    else:
        train_params = {
                'validate_first': False,
                'data_params': train_data_param,
                'queue_params': train_queue_params,
                'thres_loss': float('Inf'),
                'num_steps': 1900000 * NUM_BATCHES_PER_EPOCH,
                }

    if args.with_rep==1:
        train_params['targets'] = loss_rep_targets

    if args.instance_task:
        # report loss_model and loss_noise
        train_params['targets'] = {
                'func': utils.rep_losses,
                'devices': model_params['devices'],
                }
        train_params['targets'].update(loss_func_kwargs)

    if args.valid_first==1:
        if args.tpu_task is None:
            train_params['validate_first'] = True
        else:
            train_params['tpu_validate_first'] = True

    optimizer_class = tf.train.MomentumOptimizer
    clip_flag = args.withclip==1

    optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': optimizer_class,
            'clip': clip_flag,
            'momentum': .9,
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
                'use_nesterov': True,
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
    if args.whichopt==6:
        optimizer_params = {
                'func': optimizer.ClipOptimizer,
                'optimizer_class': tf.train.GradientDescentOptimizer,
                'clip': clip_flag,
                }
    if args.whichopt==7:
        optimizer_params = {
                'func': optimizer.ClipOptimizer,
                'optimizer_class': optimizer_class,
                'clip': clip_flag,
                'momentum': .999,
                }

    if args.tpu_task:
        optimizer_params['clip_min'] = -args.clip_num
        optimizer_params['clip_max'] = args.clip_num

    feats_target = {
            'func': utils.save_features,
            'num_to_save': 5,
            'targets': val_target,
            'cfg_dataset': cfg_dataset,
            'depth_norm': args.depth_norm,
            'normalloss': args.normalloss,
            'depthnormal': args.depthnormal,
            'extra_feat': args.extra_feat,
            }

    col_target = {
            'func': utils.gpu_col_feat,
            }

    feats_val_data_param = copy.deepcopy(val_data_param)

    if not args.featdconfig is None:
        dataconfig = args.featdconfig
        if not os.path.exists(dataconfig):
            dataconfig = os.path.join('dataset_configs', dataconfig)
        cfg_dataset_feat = utils.postprocess_config(
                json.load(open(dataconfig))
                )
        val_target_feat = utils.get_val_target(cfg_dataset_feat)

        feats_target['target'] = val_target_feat
        feats_target['cfg_dataset'] = cfg_dataset_feat
        feats_val_data_param['cfg_dataset'] = cfg_dataset_feat

    feats_val_param = {
            'data_params': feats_val_data_param,
            'queue_params': val_queue_params,
            'targets': feats_target,
            'num_steps': 10,
            'agg_func': utils.mean_losses_keep_rest,
            }

    col_val_param = {
            'data_params': topn_val_data_param,
            'queue_params': val_queue_params,
            'targets': col_target,
            'num_steps': 1,
            'agg_func': utils.mean_losses_keep_rest,
            }
    feats_train_param = {
            'data_params': train_data_param,
            'queue_params': val_queue_params,
            'targets': feats_target,
            'num_steps': 10,
            'agg_func': utils.mean_losses_keep_rest,
            }
    
    if args.tpu_task:
        topn_val_param = {
                'data_params': topn_val_data_param,
                'targets': loss_rep_targets,
                'num_steps': val_step_num
                }
    else:
        topn_val_param = {
            'data_params': topn_val_data_param,
            'queue_params': val_queue_params,
            'targets': loss_rep_targets,
            'num_steps': val_step_num,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': utils.online_agg
            }    

    extra_loss_func_kwargs = {
            'label_norm': args.label_norm,
            'top_or_loss': 1,
            }
    extra_loss_func_kwargs.update(loss_rep_targets_kwargs)

    validation_params = {
            'topn': topn_val_param,
            }
    if args.with_feat==1:
        validation_params['feats'] = feats_val_param

    if args.col_image==1:
        validation_params['color'] = col_val_param

    if args.with_train==1:
        validation_params['topn_train'] = topn_train_param
        validation_params['feats_train'] = feats_train_param
    
    if args.validation_skip==1:
        validation_params = {}

    if args.sm_train_or_test==0 and args.sm_gen_features==0:
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

        if not args.minibatch is None:
            params['train_params']['minibatch_size'] = args.minibatch
    else:
        params = {
                'save_params': save_params,
                'load_params': load_params,
                'model_params': model_params,
                'validation_params': validation_params,
                }
        if args.sm_gen_features==1:
            params['dont_run'] = True

    return params


def main():
    parser = get_parser()
    args = parser.parse_args()
    args = load_setting(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if len(args.innerargs)==0:
        params = get_params_from_arg(args)
        if not params is None:
            if args.sm_gen_features==0:
                if args.sm_train_or_test==0:
                    base.train_from_params(**params)
                else:
                    base.test_from_params(**params)
            else:
                test_args = base.test_from_params(**params)
                dbinterface = test_args['dbinterface'][0]
                sess = dbinterface.sess
                queues = test_args['queues'][0]
                validation_targets = test_args['validation_targets'][0]
                coord, threads = base.start_queues(sess)
                print(type(validation_targets))
    else:
        params = {
                'save_params': [],
                'load_params': [],
                'model_params': [],
                'train_params': None,
                'loss_params': [],
                'learning_rate_params': [],
                'optimizer_params': [],
                'log_device_placement': False,
                'validation_params': [],
                'skip_check': True,
                }

        list_names = [
                "save_params", "load_params", "model_params",
                "validation_params", "loss_params", "learning_rate_params",
                "optimizer_params"
                ]

        for curr_arg in args.innerargs:
            args = parser.parse_args(curr_arg.split())
            args = load_setting(args)
            curr_params = get_params_from_arg(args)
            for tmp_key in list_names:
                params[tmp_key].append(curr_params[tmp_key])
            params['train_params'] = curr_params['train_params']
        print(params)
        base.train_from_params(**params)

if __name__=='__main__':
    main()
