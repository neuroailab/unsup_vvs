import json


def add_cfg_kwargs(args):
    cfg_kwargs = {
            'ignorebname_new': 0,
            'add_batchname': '_rp',
            'pathconfig': '../combine_pred/network_configs/rp_resnet18.cfg',
            }
    args.cfg_kwargs = json.dumps(cfg_kwargs)
    args.prep_type = 'only_mean'
    return args


def rp_seed0(args):
    args = add_cfg_kwargs(args)
    step_num = 1181162
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/rp_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-rp-seed0'
    return args


def rp_seed1(args):
    args = rp_seed0(args)
    step_num = 1080972
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/rp_res18_s1/model.ckpt-%i' % step_num
    args.identifier = 'tpu-rp-seed1'
    return args


def rp_seed2(args):
    args = rp_seed0(args)
    step_num = 1020918
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/rp_res18_s2/model.ckpt-%i' % step_num
    args.identifier = 'tpu-rp-seed2'
    return args


def rp_res50(args):
    args.prep_type = 'only_mean'
    step_num = 2522268
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/rp_resnet50/model.ckpt-%i' % step_num
    cfg_kwargs = {
            'ignorebname_new': 0,
            'add_batchname': '_rp',
            'pathconfig': '../combine_pred/network_configs/rp_resnet50.cfg',
            }
    args.cfg_kwargs = json.dumps(cfg_kwargs)
    args.identifier = 'tpu-rp-res50-seed0'
    args.layers = ','.join(['encode_%i' % i for i in range(1, 18)]) # no first conv layer
    return args
