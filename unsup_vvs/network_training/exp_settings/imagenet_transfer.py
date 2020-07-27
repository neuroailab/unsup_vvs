import exp_settings.shared_settings as shared_sts
import cmd_parser
from models.config_parser import get_network_cfg
import pdb
from models.network_cfg_scripts.shared_funcs import get_category


def transfer_basic(args):
    args = shared_sts.basic_setting(args)
    args = shared_sts.bs_256_less_save_setting(args)
    args.dataconfig = 'dataset_config_image.cfg'
    args.nport = 26001
    args.dbname = 'cb_image_trans'
    args.collname = 'res18'
    args.trainable_scope = 'category_trans'
    args.drop_global_step = True
    args.lr_boundaries = '200000,500000,700000'
    args.weight_decay = 1e-4
    args.train_num_steps = 801000
    return args


def load_prev_func(args, load_func):
    train_args = cmd_parser.get_parser().parse_args([])
    train_args.load_setting_func = load_func
    train_args = cmd_parser.load_setting(train_args)

    args.loadport = train_args.nport
    args.load_dbname = train_args.dbname
    args.load_collname = train_args.collname
    args.loadexpId = train_args.expId
    return args, train_args


def trans_network_cfg(train_args):
    old_config = get_network_cfg(train_args)
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "category_trans"],
            "imagenet_order": ["encode", "category_trans"],
            }
    ret_cfg["encode"] = old_config["encode"]
    if 'as_output' in ret_cfg["encode"]:
        ret_cfg["encode"].pop('as_output')
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    ret_cfg["category_trans"] = {
            "as_output":1,
            "input": "encode_%i" % num_layers_enc,
            1: {"fc": {"num_features": 1000, "output": 1}}
            }
    ret_cfg["category_trans_depth"] = 1
    return ret_cfg


def la_s0(args):
    args = transfer_basic(args)

    load_func = 'combine_irla_others.res18_la'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args.expId = 'la_s0_fx'
    args.init_lr = 0.01
    return args


def ir_s0(args):
    args = transfer_basic(args)

    load_func = 'combine_irla_others.res18_ir'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args.expId = 'ir_s0'
    args.init_lr = 0.01
    return args


def la_s1(args):
    args = transfer_basic(args)

    load_func = 'combine_irla_others.res18_la_s1'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args.expId = 'la_s1'
    args.init_lr = 0.01
    return args


def ir_s1(args):
    args = transfer_basic(args)

    load_func = 'combine_irla_others.res18_ir_s1'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args.expId = 'ir_s1'
    args.init_lr = 0.01
    return args


def la_s2(args):
    args = transfer_basic(args)

    load_func = 'combine_irla_others.res18_la_s2'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args.expId = 'la_s2'
    args.init_lr = 0.01
    return args


def ir_s2(args):
    args = transfer_basic(args)

    load_func = 'combine_irla_others.res18_ir_s2'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args.expId = 'ir_s2'
    args.init_lr = 0.01
    return args


def clear_load_params(args):
    args.loadport = None
    args.load_dbname = None
    args.load_collname = None
    args.loadexpId = None
    return args


def untrn_s0(args):
    args = transfer_basic(args)

    load_func = 'cate_res18_exp0'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args = clear_load_params(args)
    args.expId = 'untrn_s0'
    args.init_lr = 0.01
    return args


def untrn_s1(args):
    args = transfer_basic(args)

    load_func = 'cate_res18_exp1'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args = clear_load_params(args)
    args.expId = 'untrn_s1'
    args.init_lr = 0.01
    return args


def untrn_s2(args):
    args = transfer_basic(args)

    load_func = 'cate_res18_exp2'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args = clear_load_params(args)
    args.expId = 'untrn_s2'
    args.init_lr = 0.01
    return args


def ae_s0(args):
    args = transfer_basic(args)

    load_func = 'other_tasks.res18_AE_imagenet'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args.expId = 'ae_s0'
    args.init_lr = 0.01
    return args


def ae_s1(args):
    args = transfer_basic(args)

    load_func = 'other_tasks.res18_AE_imagenet_seed1'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args.expId = 'ae_s1'
    args.init_lr = 0.01
    return args


def ae_s2(args):
    args = transfer_basic(args)

    load_func = 'other_tasks.res18_AE_imagenet_seed2'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args.expId = 'ae_s2'
    args.init_lr = 0.01
    return args


def cpc_s0(args):
    args = transfer_basic(args)

    load_func = 'other_tasks.res18_cpc_imagenet_tpu'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args = clear_load_params(args)
    step_num = 130 * 10010
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/cpc/model.ckpt-%i' % step_num
    args.expId = 'cpc_s0'
    args.init_lr = 0.01
    return args


def cpc_s1(args):
    args = transfer_basic(args)

    load_func = 'other_tasks.res18_cpc_imagenet_tpu'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args = clear_load_params(args)
    step_num = 130 * 10010
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/cpc_seed1/model.ckpt-%i' % step_num
    args.expId = 'cpc_s1'
    args.init_lr = 0.01
    return args


def cpc_s2(args):
    args = transfer_basic(args)

    load_func = 'other_tasks.res18_cpc_imagenet_tpu'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args = clear_load_params(args)
    step_num = 130 * 10010
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/cpc_seed2/model.ckpt-%i' % step_num
    args.expId = 'cpc_s2'
    args.init_lr = 0.01
    return args


def depth_network_cfg_setting(args):
    train_args = cmd_parser.get_parser().parse_args([])
    train_args.pathconfig = 'depth_resnet18_up4.cfg'
    args.network_func = lambda: trans_network_cfg(train_args)
    return args


def depth_s0(args):
    args = transfer_basic(args)

    args = depth_network_cfg_setting(args)
    args = clear_load_params(args)
    step_num = 2982682
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/depth_res18/model.ckpt-%i' % step_num
    args.ignorebname_new = 0
    args.add_batchname = '_pbrnet'
    args.no_prep = 1

    args.expId = 'depth_s0'
    args.init_lr = 0.01
    return args


def depth_s1(args):
    args = depth_s0(args)

    args.expId = 'depth_s1'
    step_num = 3583222
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/depth_res18_s1/model.ckpt-%i' % step_num
    return args


def depth_s2(args):
    args = depth_s0(args)

    args.expId = 'depth_s2'
    step_num = 3102790
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/depth_res18_s2/model.ckpt-%i' % step_num
    return args


def color_s0(args):
    args = transfer_basic(args)

    args = depth_network_cfg_setting(args)
    args = clear_load_params(args)
    step_num = 5605040
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.ignorebname_new = 0
    args.add_batchname = '_colorization'
    args.color_dp_tl = 1
    args.no_prep = 1

    args.expId = 'color_s0'
    args.init_lr = 0.01
    return args


def color_s1(args):
    args = color_s0(args)

    args.expId = 'color_s1'
    step_num = 4303870
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18_s1/model.ckpt-%i' % step_num
    return args


def color_s2(args):
    args = color_s0(args)

    args.expId = 'color_s2'
    step_num = 5184662
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18_s2/model.ckpt-%i' % step_num
    return args


def rp_s0(args):
    args = transfer_basic(args)

    train_args = cmd_parser.get_parser().parse_args([])
    train_args.pathconfig = 'rp_resnet18.cfg'
    args.network_func = lambda: trans_network_cfg(train_args)

    args = clear_load_params(args)
    step_num = 1181162
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/rp_res18/model.ckpt-%i' % step_num
    args.ignorebname_new = 0
    args.add_batchname = '_rp'
    args.rp_dp_tl = 1
    args.no_prep = 1

    args.expId = 'rp_s0'
    args.init_lr = 0.01
    return args


def rp_s1(args):
    args = rp_s0(args)

    args.expId = 'rp_s1'
    step_num = 1080972
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/rp_res18_s1/model.ckpt-%i' % step_num
    return args


def rp_s2(args):
    args = rp_s0(args)

    args.expId = 'rp_s2'
    step_num = 1020918
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/rp_res18_s2/model.ckpt-%i' % step_num
    return args


def la_saycam(args):
    args = transfer_basic(args)

    load_func = 'combine_irla_others.res18_la_saycam'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args.expId = 'la_saycam'
    args.init_lr = 0.01
    return args


def la_saycam_all_125(args):
    args = transfer_basic(args)

    load_func = 'combine_irla_others.res18_la_saycam_all_125'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args.expId = 'la_saycam_all_125'
    args.init_lr = 0.01
    return args


def la_saycam_all_25(args):
    args = transfer_basic(args)

    load_func = 'combine_irla_others.res18_la_saycam_all_25'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args.expId = 'la_saycam_all_25'
    args.init_lr = 0.01
    return args


def ir_saycam_all_25(args):
    args = transfer_basic(args)

    load_func = 'combine_irla_others.res18_ir_saycam_all_25'
    args, train_args = load_prev_func(args, load_func)
    args.network_func = lambda: trans_network_cfg(train_args)
    args.expId = 'ir_saycam_all_25'
    args.init_lr = 0.01
    return args


def simclr(args):
    args = transfer_basic(args)
    args.loadport = 26001
    args.namefunc = 'simclr_func'
    args.ckpt_file = '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18/model.ckpt-311748'
    args.expId = 'simclr'
    args.init_lr = 0.01
    args.resnet_prep = True
    args.resnet_prep_size = True
    return args


def simclr_prep(args):
    args = transfer_basic(args)
    args.loadport = 26001
    args.namefunc = 'simclr_func'
    #args.ckpt_file = '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18/model.ckpt-311748'
    args.expId = 'simclr_prep'
    args.init_lr = 0.01
    return args


def simclr_prep_seed1(args):
    args = transfer_basic(args)
    args.loadport = 26001
    args.namefunc = 'simclr_func'
    args.ckpt_file = '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18_2/model.ckpt-311748'
    args.expId = 'simclr_prep_seed1'
    args.init_lr = 0.01
    return args


def simclr_prep_seed2(args):
    args = transfer_basic(args)
    args.loadport = 26001
    args.namefunc = 'simclr_func'
    args.ckpt_file = '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18_3/model.ckpt-311748'
    args.expId = 'simclr_prep_seed2'
    args.init_lr = 0.01
    return args


def set_load_param_dict(args):
    import sys, os
    sys.path.append(
            os.path.abspath('../neural_fit/brainscore_mask'))
    import tf_model_loader
    import tensorflow as tf
    import json
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
    var_dict = {}
    for each_var in var_shapes:
        if ('prednet' in each_var) \
                and ('layer' in each_var) \
                and ('Adam' not in each_var):
            var_dict[each_var] = each_var.strip('__GPU0__/')
    args.load_param_dict = json.dumps(var_dict)
    return args


def prednet_kinetics_A(args):
    args = transfer_basic(args)

    args.loadport = 26001
    args.load_dbname = 'vd_unsup_fx'
    args.load_collname = 'prednet_fx3'
    args.loadexpId = 'infant_l9'
    args.loadstep = 100000
    #args = set_load_param_dict(args)

    args.namefunc = 'prednet:prednet_l9:A_9'
    args.expId = 'prednet_kinetics_l9_A'
    args.init_lr = 0.01
    args.validation_skip = True
    return args


def prednet_kinetics_E(args):
    args = prednet_kinetics_A(args)
    args.namefunc = 'prednet:prednet_l9:E_9'
    args.expId = 'prednet_kinetics_l9_E'
    return args


def prednet_kinetics_R(args):
    args = prednet_kinetics_A(args)
    args.namefunc = 'prednet:prednet_l9:R_9'
    args.expId = 'prednet_kinetics_l9_R'
    return args


def prednet_kinetics_Ahat(args):
    args = prednet_kinetics_A(args)
    args.namefunc = 'prednet:prednet_l9:Ahat_9'
    args.expId = 'prednet_kinetics_l9_Ahat'
    return args


def prednet_kinetics_l3(args):
    args = prednet_kinetics_A(args)
    args.loadexpId = 'kinetics'
    args.namefunc = 'l3_prednet'
    args.expId = 'prednet_kinetics_l3'
    return args


def prednet_kinetics_l3_layer2(args):
    args = prednet_kinetics_A(args)
    args.loadexpId = 'kinetics'
    args.namefunc = 'l3_prednet_layer2'
    args.expId = 'prednet_kinetics_l3_layer2'
    return args


def prednet_kinetics_l3_tpu_raw(args):
    args = prednet_kinetics_A(args)
    args.loadexpId = 'kinetics'
    args.namefunc = 'l3_prednet'

    args.expId = 'prednet_kinetics_l3_tpu_raw'
    args.nport = 27001
    args.cacheDirPrefix = 'gs://cx_visualmaster/'
    args.tpu_task = 'multi_imagenet'
    return args


def prednet_kinetics_l3_tpu(args):
    args = prednet_kinetics_l3_tpu_raw(args)
    args.expId = 'prednet_kinetics_l3_tpu'
    return args


def prednet_kinetics_l3_layer2_tpu(args):
    args = prednet_kinetics_l3_tpu_raw(args)
    args.namefunc = 'l3_prednet_layer2'
    args.expId = 'prednet_kinetics_l3_layer2_tpu'
    return args
