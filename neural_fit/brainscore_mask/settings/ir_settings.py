import json


def ir_seed0(args):
    args.load_step = 250 * 10010
    args.setting_name = 'combine_irla_others.res18_ir'
    return args


def ir_seed1(args):
    args.load_step = 250 * 10010
    args.setting_name = 'combine_irla_others.res18_ir_s1'
    return args


def ir_seed2(args):
    args.load_step = 250 * 10010
    args.setting_name = 'combine_irla_others.res18_ir_s2'
    return args


def ir_saycam(args):
    args.load_step = 320 * 10010
    args.setting_name = 'combine_irla_others.res18_ir_saycam'
    args.layers = ','.join(['encode_%i' % i for i in range(1, 10)]) # no first conv layer
    return args


def ir_saycam_125(args):
    args.load_step = 300 * 10010
    args.setting_name = 'combine_irla_others.res18_ir_saycam_all_125'
    args.layers = ','.join(['encode_%i' % i for i in range(1, 10)]) # no first conv layer
    return args


def ir_saycam_25(args):
    args.load_step = 460 * 10010
    args.setting_name = 'combine_irla_others.res18_ir_saycam_all_25'
    args.layers = ','.join(['encode_%i' % i for i in range(1, 10)]) # no first conv layer
    return args


def ir_res50(args):
    cfg_kwargs = {
            'inst_resnet_size': 50,
            }
    args.cfg_kwargs = json.dumps(cfg_kwargs)
    args.model_type = 'inst_model'
    args.load_port = 27009
    args.load_dbname = 'instance_task'
    args.load_colname = 'control'
    args.load_expId = 'full50'
    args.load_step = 260 * 10009
    args.layers = ','.join(['encode_%i' % i for i in range(2, 19)]) # no first conv layer
    return args
