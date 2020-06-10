import exp_settings.shared_settings as shared_sts
import cmd_parser
from models.config_parser import get_network_cfg
import pdb
from models.network_cfg_scripts.shared_funcs import get_category
from exp_settings.imagenet_transfer import load_prev_func, clear_load_params


def transfer_basic(args):
    args = shared_sts.basic_setting(args)
    args = shared_sts.bs_256_less_save_setting(args)
    args.dataconfig = 'dataset_config_image.cfg'
    args.nport = 26001
    args.dbname = 'cb_image_mid_trans'
    args.collname = 'res18'
    args.trainable_scope = 'category_trans'
    args.drop_global_step = True
    args.lr_boundaries = '150000,300000,450000'
    args.weight_decay = 1e-4
    args.train_num_steps = 501000
    return args


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
            1: {"fc": {"num_features": 1000}},
            2: {"fc": {"num_features": 1000, "output": 1}},
            }
    ret_cfg["category_trans_depth"] = 2
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
    #args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/cpc/model.ckpt-%i' % step_num
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
    #args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/cpc_seed1/model.ckpt-%i' % step_num
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
    #args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/cpc_seed2/model.ckpt-%i' % step_num
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
    #args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/depth_res18/model.ckpt-%i' % step_num
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
    #args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/depth_res18_s1/model.ckpt-%i' % step_num
    return args


def depth_s2(args):
    args = depth_s0(args)

    args.expId = 'depth_s2'
    step_num = 3102790
    #args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/depth_res18_s2/model.ckpt-%i' % step_num
    return args


def color_s0(args):
    args = transfer_basic(args)

    args = depth_network_cfg_setting(args)
    args = clear_load_params(args)
    step_num = 5605040
    #args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
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
    #args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18_s1/model.ckpt-%i' % step_num
    return args


def color_s2(args):
    args = color_s0(args)

    args.expId = 'color_s2'
    step_num = 5184662
    #args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18_s2/model.ckpt-%i' % step_num
    return args


def rp_s0(args):
    args = transfer_basic(args)

    train_args = cmd_parser.get_parser().parse_args([])
    train_args.pathconfig = 'rp_resnet18.cfg'
    args.network_func = lambda: trans_network_cfg(train_args)

    args = clear_load_params(args)
    step_num = 1181162
    #args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/rp_res18/model.ckpt-%i' % step_num
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
    #args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/rp_res18_s1/model.ckpt-%i' % step_num
    return args


def rp_s2(args):
    args = rp_s0(args)

    args.expId = 'rp_s2'
    step_num = 1020918
    #args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/rp_res18_s2/model.ckpt-%i' % step_num
    return args


def simclr_prep(args):
    args = transfer_basic(args)
    args.loadport = 26001
    args.namefunc = 'simclr_func_mid'
    args.ckpt_file = '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18/model.ckpt-311748'
    args.expId = 'simclr_prep'
    args.init_lr = 0.01
    return args


def simclr_prep_seed1(args):
    args = transfer_basic(args)
    args.loadport = 26001
    args.namefunc = 'simclr_func_mid'
    args.ckpt_file = '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18_2/model.ckpt-311748'
    args.expId = 'simclr_prep_seed1'
    args.init_lr = 0.01
    return args


def simclr_prep_seed2(args):
    args = transfer_basic(args)
    args.loadport = 26001
    args.namefunc = 'simclr_func_mid'
    args.ckpt_file = '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18_3/model.ckpt-311748'
    args.expId = 'simclr_prep_seed2'
    args.init_lr = 0.01
    return args
