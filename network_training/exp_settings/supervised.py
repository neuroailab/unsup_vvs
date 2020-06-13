from exp_settings.shared_settings import basic_res18


def cate_res18_exp0(args):
    args = basic_res18(args)
    args.nport = 27007
    args.dbname = 'pub_cate'
    args.collname = 'res18'
    args.expId = 'exp_seed0'
    args.resnet_prep = True
    args.resnet_prep_size = True
    args.lr_boundaries = "160000,310000,460000"
    args.train_num_steps = 510000
    args.seed = 0
    return args


def cate_res18_exp1(args):
    args = cate_res18_exp0(args)
    args.expId = 'exp_seed1'
    args.seed = 1
    return args


def cate_res18_exp2(args):
    args = cate_res18_exp0(args)
    args.expId = 'exp_seed2'
    args.seed = 2
    return args
