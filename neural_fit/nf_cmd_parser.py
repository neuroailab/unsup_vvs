import argparse
import nf_saved_setting
import importlib
from argparse import Namespace
import pdb

import sys
sys.path.append('../combine_pred/')
import cmd_parser


def add_general_settings(parser):
    # General settings
    parser.add_argument(
            '--nport', default=27009, type=int, action='store',
            help='Port number of mongodb')
    parser.add_argument(
            '--expId', default="neural_test", type=str, action='store',
            help='Name of experiment id')
    parser.add_argument(
            '--dbname', default='neuralfit-test', type=str, action='store',
            help='Name of experiment database to save to')
    parser.add_argument(
            '--colname', default='neuralfit', type=str, action='store',
            help='Name of experiment collection name to save to')
    parser.add_argument(
            '--loadport', default=27010, type=int, action='store',
            help='Port number of mongodb for loading')
    parser.add_argument(
            '--loadexpId', default=None,
            type=str, action='store',
            help='Name of experiment id')
    parser.add_argument(
            '--loadstep', default=None, type=int, action='store',
            help='Number of steps for loading')
    parser.add_argument(
            '--loaddbname', default='combinet-test', type=str, action='store',
            help='Name of experiment database to load from')
    parser.add_argument(
            '--loadcolname', default='combinet', type=str, action='store',
            help='Name of experiment collection name to load from')
    parser.add_argument(
            '--cacheDirPrefix', default="/mnt/fs1/chengxuz", 
            type=str, action='store',
            help='Prefix of cache directory')
    parser.add_argument(
            '--cacheDirName', default="neural_test", type=str, action='store',
            help='Name of cache directory')
    parser.add_argument(
            '--ckpt_file', default=None, type=str, action='store',
            help='Whether load the model from ckpt file')
    parser.add_argument(
            '--train_num_steps', default=100000000, type=int, action='store',
            help='Train until this step')

    # Saving metric
    parser.add_argument(
            '--fre_valid', default=60, type=int, action='store',
            help='Frequency of the validation')
    parser.add_argument(
            '--fre_metric', default=60, type=int, action='store',
            help='Frequency of the saving metrics')
    parser.add_argument(
            '--fre_filter', default=1000000000000, type=int, action='store',
            help='Frequency of the saving filters')

    # GPU related
    parser.add_argument(
            '--gpu', default='0', type=str, action='store',
            help='Availabel GPUs')
    parser.add_argument(
            '--n_gpus', default=None, type=int, action='store',
            help='Number of GPUs to use, default is None, to use length in gpu')
    parser.add_argument(
            '--gpu_offset', default=0, type=int, action='store',
            help='Offset of gpu index')
    parser.add_argument(
            '--minibatch', default=None, type=int, action='store',
            help='Minibatch to use, default to be None, not using')

    # Pre-configed settings
    parser.add_argument(
	    '--load_setting_func', default=None, type=str, action='store',
            help='Saved setting function')
    parser.add_argument(
	    '--load_train_setting_func', default=None, type=str, action='store',
            help='Saved setting function used during training')
    parser.add_argument(
	    '--pat_func', default='v1v4it_wd_swd', type=str, action='store',
            help='Used in pat_nf_fit as func name')
    parser.add_argument(
	    '--pat_func_args', default=None, type=str, action='store',
            help='Json string as args for the pat func')

    return parser


def add_data_settings(parser):
    # Data related
    parser.add_argument(
            '--which_split', default="split_0", type=str, action='store',
            help='Which split to use in dataset')
    parser.add_argument(
            '--image_prefix', 
            default="/mnt/fs0/datasets/neural_data/img_split/", 
            type=str, action='store',
            help='Prefix for images')
    parser.add_argument(
            '--neuron_resp_prefix', default="/mnt/fs1/Dataset/neural_resp/", 
            type=str, action='store',
            help='Prefix for neural responses')
    parser.add_argument(
            '--train_or_test', default=0, type=int, action='store',
            help='Whether use SGD for training (train_from_params),'\
                    + ' or just PLS regression (test_from_params)')
    parser.add_argument(
            '--ave_or_10ms', default=0, type=int, action='store',
            help='Take data from averaged neural responses or 10ms data')
    parser.add_argument(
            '--f10ms_time', default='6,7,8,9,10,11,12,13,14', 
            type=str, action='store',
            help='Time steps to be fitted')
    parser.add_argument(
            '--val_on_train', default=0, type=int, action='store',
            help='Whether validating on train set')
    parser.add_argument(
            '--deepmind', default=0, type=int, action='store',
            help='Whether using deepmind features')
    parser.add_argument(
            '--use_dataset_inter', action='store_true',
            help='Using dataset interface')
    parser.add_argument(
            '--objectome_zip', action='store_true',
            help='Use objectome zip png files')
    parser.add_argument(
            '--obj_dataset_type', action='store', 
            type=str, default='objectome',
            help='Dataset type for objectome data provider')
    parser.add_argument(
            '--cadena_im_size', action='store', 
            type=int, default=40,
            help='Image size for Tolias Cadena data provider')
    parser.add_argument(
            '--dataset_type', action='store',
            type=str, default='hvm')
    parser.add_argument(
            '--v1v2_folder', default='/mnt/fs4/chengxuz/v1v2_related/tfrs/', 
            type=str, action='store',
            help='Tfrecords directory for v1v2 data')
    parser.add_argument(
            '--deepcluster', default=None, type=str, action='store',
            help='Use deep cluster features')
    parser.add_argument(
            '--h5py_data_loader', action='store_true',
            help='Use h5py ventral visual data loader')

    # Preprocessing related
    parser.add_argument(
            '--no_prep', default=1, type=int, action='store',
            help='Avoid the scaling in model function or not')
    parser.add_argument(
            '--rp_sub_mean', default=0, type=int, action='store',
                help = 'when test on RP model, need to subtract the mean')
    parser.add_argument(
            '--div_std', default=0, type=int, action='store',
                help = 'whether divide standard divation')
    parser.add_argument(
            '--color_prep', default=0, type=int, action='store',
                help = 'when test on color model, need to get l channel')
    parser.add_argument(
            '--combine_col_rp', default=0, type=int, action='store',
                help = 'whether input 3 channel grayscale images')
    parser.add_argument(
            '--input_mode', default='rgb', type=str, action='store',
            help='Input mode for instance')

    return parser


def add_feature_settings(parser):
    # Generate features related
    parser.add_argument(
            '--gen_features', default=0, type=int, action='store',
            help='Generating features rather than do the neural fitting')
    parser.add_argument(
            '--gen_hdf5path', 
            default='/data2/chengxuz/vm_response/response.hdf5', 
            type=str, action='store',
            help='Path to store the exported features')
    parser.add_argument(
            '--gen_features_onval', default=0, type=int, action='store',
            help='Whether generating features on validation set')

    return parser


def add_network_cfg_settings(parser):
    parser.add_argument(
            '--pathconfig', default="combinet_nbn_all_general.cfg", 
            type=str, action='store',
            help='Path to config file')
    parser.add_argument(
	    '--network_func', default=None, 
            type=str, action='store',
            help='Function name stored in network_cfg_funcs')
    parser.add_argument(
	    '--network_func_kwargs', default=None, 
            type=str, action='store',
            help='Kwargs for function stored in network_cfg_funcs')
    parser.add_argument(
            '--configdir', default="../combine_pred/network_configs", 
            type=str, action='store',
            help='Path to config file')
    return parser


def add_network_settings(parser):
    parser = add_network_cfg_settings(parser)
    # Network related
    parser.add_argument(
            '--seed', default=0, type=int, action='store',
            help='Random seed for model')
    parser.add_argument(
            '--namefunc', default="combinet_neural_fit", 
            type=str, action='store',
            help='Name of function to build the network')
    parser.add_argument(
            '--fix_pretrain', default=1, type=int, action='store',
            help='Whether fix the pretrained weights')
    parser.add_argument(
            '--v4_nodes', default=None, type=str, action='store',
            help='Nodes to fit for V4')
    parser.add_argument(
            '--it_nodes', default=None, type=str, action='store',
            help='Nodes to fit for IT')
    parser.add_argument(
            '--cache_filter', default=1, type=int, action='store',
            help='Whether cache the pretrained weights as tf tensors')
    parser.add_argument(
            '--in_conv_form', default=0, type=int, action='store',
            help='Whether use conv or fc form')
    parser.add_argument(
            '--partnet_train', default=0, type=int, action='store',
            help='Use stored batch norm means/vars, or compute it online')
    parser.add_argument(
            '--sm_resnetv2', default=0, type=int, action='store',
            help="Whether use resnet v2")  # sm_add
    parser.add_argument(
            '--mean_teacher', default=0, type=int, action='store',
            help="Mean teacher mode or not, default is 0, no")
    parser.add_argument(
            '--ema_decay', default=0.9997, type=float, action='store',
            help='Teacher decay used in mean teacher setting')
    parser.add_argument(
            '--ema_zerodb', default=0, type=float, action='store',
            help='Whether zero debias the ema, default is 0 (none)')
    parser.add_argument(
            '--inst_model', default=None, type=str, action='store',
            help="Use instance models, if not none, "\
                    + "will treat as get_all_layers")
    parser.add_argument(
            '--inst_res_size', default=18, type=int, action='store')
    parser.add_argument(
            '--spatial_select', default=None, type=str, action='store',
            help='Spatial subselecting parameter for each layer, '\
                    + '"stride,size"')
    parser.add_argument(
            '--convrnn_model', action='store_true',
            help="Use convrnn models")

    parser.add_argument(
            '--tpu_flag', default=0, type=int, action='store',
            help='The difference between gpu and tpu batchnorm')
    return parser


def add_train_settings(parser):
    # Training related
    parser.add_argument(
            '--batchsize', default=64, type=int, action='store',
            help='Batch size')
    parser.add_argument(
            '--init_type', default='xavier', type=str, action='store',
            help='Init type')
    parser.add_argument(
            '--weight_decay', default=None, type=float, action='store',
            help='Weight decay')
    parser.add_argument(
            '--weight_decay_type', default='l2', type=str, action='store',
            help='Weight decay type, l2 (default) or l1')
    parser.add_argument(
            '--random_sample', default=None, type=int, action='store',
            help='Whether do random sampling to fc layers')
    parser.add_argument(
            '--random_sample_seed', default=0, type=int, action='store',
            help='Whether do random sampling to fc layers')
    parser.add_argument(
            '--ignorebname_new', default=1, type=int, action='store',
            help='Whether ignore the batch name in conv and resblock')
    parser.add_argument(
            '--batch_name', default='_imagenet', type=str, action='store',
            help='Suffix name for the batch norm')

    ## Learning rate, optimizers
    parser.add_argument(
            '--init_lr', default=.01, type=float, action='store',
            help='Init learning rate')
    parser.add_argument(
            '--whichopt', default=0, type=int, action='store',
            help='Choice of the optimizer, 0 means momentum, 1 means Adam')
    parser.add_argument(
            '--adameps', default=0.1, type=float, action='store',
            help='Epsilon for adam, only used when whichopt is 1')
    parser.add_argument(
            '--adambeta1', default=0.9, type=float, action='store',
            help='Beta1 for adam, only used when whichopt is 1')
    parser.add_argument(
            '--adambeta2', default=0.999, type=float, action='store',
            help='Beta2 for adam, only used when whichopt is 1')
    parser.add_argument(
            '--withclip', default=1, type=int, action='store',
            help='Whether do clip')
    parser.add_argument(
            '--img_out_size', default=224, type=int, action='store',
            help='Size of input image for tfrecords data providers')
    parser.add_argument(
            '--img_crop_size', default=None, type=int, action='store',
            help='Center crop size from the original image if needed')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to fit to neural data')

    parser = add_general_settings(parser)
    parser = add_data_settings(parser)
    parser = add_feature_settings(parser)
    parser = add_network_settings(parser)
    parser = add_train_settings(parser)

    return parser


def load_setting(args):
    if args.load_setting_func:
        load_setting_func = args.load_setting_func

        if '.' in load_setting_func:
            all_paths = args.load_setting_func.split('.')
            module_name = '.'.join(['nf_exp_settings'] + all_paths[:-1])
            load_setting_module = importlib.import_module(module_name)
            load_setting_func = all_paths[-1]
            setting_func = getattr(load_setting_module, load_setting_func)
        else:
            setting_func = getattr(nf_saved_setting, load_setting_func)
        args = setting_func(args)

    if args.load_train_setting_func:
        train_args = Namespace()
        train_args.load_setting_func = args.load_train_setting_func
        train_args = cmd_parser.load_setting(train_args)

        args.input_mode = getattr(train_args, 'input_mode', None) \
                or getattr(args, 'input_mode', 'rgb')
        args.loadport = getattr(train_args, 'nport', None) or args.loadport
        args.loaddbname = getattr(train_args, 'dbname', None) or args.loaddbname
        args.loadcolname = getattr(train_args, 'collname', None) \
                or args.loadcolname
        args.loadexpId = train_args.expId
        args.network_func = getattr(train_args, 'network_func', None) \
                or args.network_func
        args.network_func_kwargs = getattr(
                train_args, 
                'network_func_kwargs', None) \
                        or args.network_func_kwargs
        if args.network_func is None:
            args.pathconfig = getattr(train_args, 'pathconfig', None) \
                    or args.pathconfig
    return args
