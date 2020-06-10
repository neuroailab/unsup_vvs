def ae_seed0(args):
    args.load_step = 130 * 10010
    args.setting_name = 'other_tasks.res18_AE_imagenet'
    return args


def ae_seed1(args):
    args.load_step = 130 * 10010
    args.setting_name = 'other_tasks.res18_AE_imagenet_seed1'
    return args


def ae_seed2(args):
    args.load_step = 130 * 10010
    args.setting_name = 'other_tasks.res18_AE_imagenet_seed2'
    return args


def ae_wd_seed0(args):
    args.load_step = 150 * 10010
    args.setting_name = 'other_tasks.res18_AE_imagenet_wd_l2'
    return args


def ae_l1_seed0(args):
    args.load_step = 80 * 10010
    args.setting_name = 'other_tasks.res18_AE_imagenet_l1'
    return args
