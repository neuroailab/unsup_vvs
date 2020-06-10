import json


def add_cfg_kwargs(args):
    cfg_kwargs = {
            'ignorebname_new': 0,
            'add_batchname': '_pbrnet',
            'pathconfig': '../combine_pred/network_configs/depth_resnet18_up4.cfg',
            }
    args.cfg_kwargs = json.dumps(cfg_kwargs)
    args.prep_type = 'no_prep'
    return args


def depth_seed0(args):
    args = add_cfg_kwargs(args)
    step_num = 2982682
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/depth_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-depth-seed0'
    return args


def depth_seed1(args):
    args = depth_seed0(args)
    step_num = 3583222
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/depth_res18_s1/model.ckpt-%i' % step_num
    args.identifier = 'tpu-depth-seed1'
    return args


def depth_seed2(args):
    args = depth_seed0(args)
    step_num = 3102790
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/depth_res18_s2/model.ckpt-%i' % step_num
    args.identifier = 'tpu-depth-seed2'
    return args


def depth_res50(args):
    args.prep_type = 'no_prep'
    step_num = 1361224
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/depth_res50/model.ckpt-%i' % step_num
    cfg_kwargs = {
            'ignorebname_new': 0,
            'add_batchname': '_pbrnet',
            'pathconfig': '../combine_pred/network_configs/depth_resnet50_up4.cfg',
            }
    args.cfg_kwargs = json.dumps(cfg_kwargs)
    args.identifier = 'tpu-depth-res50-seed0'
    args.layers = ','.join(['encode_%i' % i for i in range(1, 18)]) # no first conv layer
    return args
