import json


def add_cfg_kwargs(args):
    cfg_kwargs = {
            'ignorebname_new': 0,
            'add_batchname': '_colorization',
            'pathconfig': '../combine_pred/network_configs/depth_resnet18_up4.cfg',
            }
    args.cfg_kwargs = json.dumps(cfg_kwargs)
    args.prep_type = 'color_prep'
    return args


def color_seed0(args):
    args = add_cfg_kwargs(args)
    step_num = 5605040
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed0'
    return args


def color_seed1(args):
    args = color_seed0(args)
    step_num = 4303870
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18_s1/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed1'
    return args


def color_seed2(args):
    args = color_seed0(args)
    step_num = 5184662
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18_s2/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed2'
    return args


def color_input(args):
    args = color_seed0(args)
    args.just_input = True
    return args


def color_seed0_ep2(args):
    args = color_seed0(args)
    step_num = 20018 * 2
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed0-ep2'
    return args


def color_seed0_ep4(args):
    args = color_seed0(args)
    step_num = 20018 * 4
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed0-ep4'
    return args


def color_seed0_ep6(args):
    args = color_seed0(args)
    step_num = 20018 * 6
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed0-ep6'
    return args


def color_seed0_ep8(args):
    args = color_seed0(args)
    step_num = 20018 * 8
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed0-ep8'
    return args


def color_seed0_ep10(args):
    args = color_seed0(args)
    step_num = 20018 * 10
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed0-ep10'
    return args


def color_seed0_ep20(args):
    args = color_seed0(args)
    step_num = 20018 * 20
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed0-ep20'
    return args


def color_seed0_ep30(args):
    args = color_seed0(args)
    step_num = 20018 * 30
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed0-ep30'
    return args


def color_seed0_ep40(args):
    args = color_seed0(args)
    step_num = 20018 * 40
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed0-ep40'
    return args


def color_seed0_ep50(args):
    args = color_seed0(args)
    step_num = 20018 * 50
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed0-ep50'
    return args


def color_seed0_ep100(args):
    args = color_seed0(args)
    step_num = 20018 * 100
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed0-ep100'
    return args


def color_seed0_ep150(args):
    args = color_seed0(args)
    step_num = 20018 * 150
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed0-ep150'
    return args


def color_seed0_ep200(args):
    args = color_seed0(args)
    step_num = 20018 * 200
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.identifier = 'tpu-color-seed0-ep200'
    return args


def color_res50(args):
    args.prep_type = 'color_prep'
    step_num = 2121908
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/col_res50/model.ckpt-%i' % step_num
    cfg_kwargs = {
            'ignorebname_new': 0,
            'add_batchname': '_colorization',
            'pathconfig': '../combine_pred/network_configs/depth_resnet50_up4.cfg',
            }
    args.cfg_kwargs = json.dumps(cfg_kwargs)
    args.identifier = 'tpu-color-res50-seed0'
    args.layers = ','.join(['encode_%i' % i for i in range(1, 18)]) # no first conv layer
    return args
