import json


def la_seed0(args):
    args.load_step = 250 * 10010
    args.setting_name = 'combine_irla_others.res18_la'
    return args


def la_seed1(args):
    args.load_step = 250 * 10010
    args.setting_name = 'combine_irla_others.res18_la_s1'
    return args


def la_seed2(args):
    args.load_step = 250 * 10010
    args.setting_name = 'combine_irla_others.res18_la_s2'
    return args


def la_seed1_ep2(args):
    load_step = 2 * 10010
    args.load_from_ckpt = '/mnt/fs4/chengxuz/brainscore_model_caches/irla_and_others/res18/la_s1/checkpoint-%i' % load_step
    args.setting_name = 'combine_irla_others.res18_la_s1'
    args.identifier = 'irla_and_others-res18-la_s1-27007-%i' % load_step
    return args


def la_seed1_ep4(args):
    load_step = 4 * 10010
    args.load_from_ckpt = '/mnt/fs4/chengxuz/brainscore_model_caches/irla_and_others/res18/la_s1/checkpoint-%i' % load_step
    args.setting_name = 'combine_irla_others.res18_la_s1'
    args.identifier = 'irla_and_others-res18-la_s1-27007-%i' % load_step
    return args


def la_seed1_ep6(args):
    load_step = 6 * 10010
    args.load_from_ckpt = '/mnt/fs4/chengxuz/brainscore_model_caches/irla_and_others/res18/la_s1/checkpoint-%i' % load_step
    args.setting_name = 'combine_irla_others.res18_la_s1'
    args.identifier = 'irla_and_others-res18-la_s1-27007-%i' % load_step
    return args


def la_seed1_ep8(args):
    load_step = 8 * 10010
    args.load_from_ckpt = '/mnt/fs4/chengxuz/brainscore_model_caches/irla_and_others/res18/la_s1/checkpoint-%i' % load_step
    args.setting_name = 'combine_irla_others.res18_la_s1'
    args.identifier = 'irla_and_others-res18-la_s1-27007-%i' % load_step
    return args


def la_seed1_ep10(args):
    load_step = 10 * 10010
    args.load_from_ckpt = '/mnt/fs4/chengxuz/brainscore_model_caches/irla_and_others/res18/la_s1/checkpoint-%i' % load_step
    args.setting_name = 'combine_irla_others.res18_la_s1'
    args.identifier = 'irla_and_others-res18-la_s1-27007-%i' % load_step
    return args


def la_seed1_ep20(args):
    args.load_step = 20 * 10010
    args.setting_name = 'combine_irla_others.res18_la_s1'
    return args


def la_seed1_ep40(args):
    args.load_step = 40 * 10010
    args.setting_name = 'combine_irla_others.res18_la_s1'
    return args


def la_seed1_ep80(args):
    args.load_step = 80 * 10010
    args.setting_name = 'combine_irla_others.res18_la_s1'
    return args


def la_seed1_ep120(args):
    args.load_step = 120 * 10010
    args.setting_name = 'combine_irla_others.res18_la_s1'
    return args


def la_seed1_ep160(args):
    args.load_step = 160 * 10010
    args.setting_name = 'combine_irla_others.res18_la_s1'
    return args


def la_seed1_ep180(args):
    args.load_step = 180 * 10010
    args.setting_name = 'combine_irla_others.res18_la_s1'
    return args


def la_seed1_ep220(args):
    args.load_step = 220 * 10010
    args.setting_name = 'combine_irla_others.res18_la_s1'
    return args


def la_seed1_ep240(args):
    args.load_step = 240 * 10010
    args.setting_name = 'combine_irla_others.res18_la_s1'
    return args


def old_la_mid(args):
    args.layers = 'mid500,mid1000,mid1500,mid2000'
    args.model_type = 'inst_model:Mid-500,1000,1500,2000'
    args.load_step = 2552295
    args.load_port = 27009
    args.load_dbname = 'instance_transfer'
    args.load_colname = 'dyn_clustering'
    args.load_expId = 'dyn_km_ormre_mid'
    return args


def la_saycam(args):
    args.load_step = 300 * 10010
    args.setting_name = 'combine_irla_others.res18_la_saycam'
    args.layers = ','.join(['encode_%i' % i for i in range(1, 10)]) # no first conv layer
    return args


def la_saycam_125(args):
    args.load_step = 330 * 10010
    args.setting_name = 'combine_irla_others.res18_la_saycam_all_125'
    args.layers = ','.join(['encode_%i' % i for i in range(1, 10)]) # no first conv layer
    return args


def la_saycam_25(args):
    args.load_step = 390 * 10010
    args.setting_name = 'combine_irla_others.res18_la_saycam_all_25'
    args.layers = ','.join(['encode_%i' % i for i in range(1, 10)]) # no first conv layer
    return args


def la_res50(args):
    cfg_kwargs = {
            'inst_resnet_size': 50,
            }
    args.cfg_kwargs = json.dumps(cfg_kwargs)
    args.model_type = 'inst_model'
    args.load_port = 27009
    args.load_dbname = 'instance_task'
    args.load_colname = 'dynamic_clustering'
    args.load_expId = 'dyn_km_50_mre10sm'
    args.load_step = 1901710
    args.layers = ','.join(['encode_%i' % i for i in range(2, 19)]) # no first conv layer
    return args
