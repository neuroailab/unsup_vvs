def cpc_seed0(args):
    args.setting_name = 'other_tasks.res18_cpc_imagenet_tpu'
    step_num = 130 * 10010
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/cpc/model.ckpt-%i' % step_num
    args.identifier = 'tpu-cpc-seed0'
    return args


def cpc_seed1(args):
    args.setting_name = 'other_tasks.res18_cpc_imagenet_tpu'
    step_num = 130 * 10010
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/cpc_seed1/model.ckpt-%i' % step_num
    args.identifier = 'tpu-cpc-seed1'
    return args


def cpc_seed2(args):
    args.setting_name = 'other_tasks.res18_cpc_imagenet_tpu'
    step_num = 130 * 10010
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/cpc_seed2/model.ckpt-%i' % step_num
    args.identifier = 'tpu-cpc-seed2'
    return args
