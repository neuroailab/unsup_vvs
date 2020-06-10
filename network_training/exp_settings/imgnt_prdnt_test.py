import exp_settings.shared_settings as shared_sts
import cmd_parser
from models.config_parser import get_network_cfg
import pdb
from models.network_cfg_scripts.shared_funcs import get_category
import exp_settings.imagenet_transfer as imagenet_transfer
import json


def get_ckpt_var_shapes(args, ckpt_file):
    import sys, os
    sys.path.append(
            os.path.abspath('../neural_fit/brainscore_mask'))
    import tf_model_loader
    import tensorflow as tf
    if ckpt_file is None:
        ckpt_file = tf_model_loader.load_model_from_mgdb(
                db=args.load_dbname,
                col=args.load_collname,
                exp=args.loadexpId,
                port=args.loadport,
                step_num=args.loadstep,
                cache_dir=tf_model_loader.DEFAULT_MODEL_CACHE_DIR,
                )
    reader = tf.train.NewCheckpointReader(ckpt_file)
    var_shapes = reader.get_variable_to_shape_map()
    return var_shapes


def set_load_param_dict(args, ckpt_file=None):
    var_shapes = get_ckpt_var_shapes(args, ckpt_file)
    var_dict = {}
    for each_var in var_shapes:
        if 'prednet' in each_var:
            var_dict[each_var] = 'validation/topn/model_0/__var_copy_0/' + each_var
    args.load_param_dict = json.dumps(var_dict)
    return args


def prednet_kinetics_l3_test(args):
    args = shared_sts.basic_setting(args)
    args = shared_sts.bs_256_less_save_setting(args)
    args.valinum = 500
    args.valbatchsize = 100
    args.dataconfig = 'dataset_config_image.cfg'
    args.nport = 26001
    args.dbname = 'cb_image_trans'
    args.expId = 'prednet_kinetics_l3_test_ep3'
    args.namefunc = 'l3_prednet'

    args.loadport = 26001
    args.load_dbname = 'cb_image_trans'
    args.load_collname = 'res18'
    args.loadexpId = 'prednet_kinetics_l3'
    args.loadstep = 5005 * 3
    ckpt_file = '/mnt/fs4/chengxuz/.tfutils/localhost:26001/cb_image_trans/combinet/prednet_kinetics_l3_test_ep3/checkpoint-15015'
    args = set_load_param_dict(args, ckpt_file)
    return args


def set_tpu_load_param_dict(args, ckpt_file=None):
    var_shapes = get_ckpt_var_shapes(args, ckpt_file)
    var_dict = {}
    for each_var in var_shapes:
        if 'prednet' in each_var:
            var_dict[each_var] = 'validation/topn/model_0/__var_copy_0/__GPU0__/' + each_var
        elif 'category_trans' in each_var:
            var_dict[each_var] = each_var
    args.load_param_dict = json.dumps(var_dict)
    return args


def set_tpu_test_basic(args):
    args = shared_sts.basic_setting(args)
    args = shared_sts.bs_256_less_save_setting(args)
    args.valinum = 500
    args.valbatchsize = 100
    args.dataconfig = 'dataset_config_image.cfg'
    args.nport = 26001
    args.dbname = 'cb_image_trans'
    args.namefunc = 'l3_prednet'
    return args


def prednet_kinetics_l3_tpu_test(args):
    args = set_tpu_test_basic(args)
    args.expId = 'prednet_kinetics_l3_tpu_test'
    ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/prednet_kinetics_l3/model.ckpt-505505'
    args = set_tpu_load_param_dict(args, ckpt_file=ckpt_file)
    args.ckpt_file = ckpt_file
    return args


def prednet_kinetics_l3_raw_tpu_test(args):
    args = set_tpu_test_basic(args)
    args.expId = 'prednet_kinetics_l3_raw_tpu_test'
    ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/prednet_kinetics_l3_raw/model.ckpt-505505'
    args = set_tpu_load_param_dict(args, ckpt_file=ckpt_file)
    args.ckpt_file = ckpt_file
    return args


def prednet_kinetics_l3_layer2_tpu_test(args):
    args = set_tpu_test_basic(args)
    args.expId = 'prednet_kinetics_l3_layer2_tpu_test'
    ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/prednet_kinetics_l3_layer2/model.ckpt-440440'
    args = set_tpu_load_param_dict(args, ckpt_file=ckpt_file)
    args.ckpt_file = ckpt_file
    args.namefunc = 'l3_prednet_layer2'
    return args
