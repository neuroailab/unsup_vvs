def untrn_seed0(args):
    args.setting_name = 'cate_res18_exp0'
    args.from_scratch = True
    return args


def untrn_seed1(args):
    args.setting_name = 'cate_res18_exp1'
    args.from_scratch = True
    return args


def untrn_seed2(args):
    args.setting_name = 'cate_res18_exp2'
    args.from_scratch = True
    return args


def untrn_fc18_seed0(args):
    args.setting_name = 'cate_fc18_exp0'
    args.from_scratch = True
    return args


def untrn_res50(args):
    args.setting_name = 'cate_res50'
    args.from_scratch = True
    args.layers = ','.join(['encode_%i' % i for i in range(1, 18)]) # no first conv layer
    return args
