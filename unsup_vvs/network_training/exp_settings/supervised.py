from exp_settings.shared_settings import basic_res18


def super_res18_s0(args):
    args = basic_res18(args)
    args.nport = 27007
    args.dbname = 'pub_super'
    args.collname = 'res18'
    args.expId = 'super_s0'
    args.resnet_prep = True
    args.resnet_prep_size = True
    args.lr_boundaries = "160000,310000,460000"
    args.train_num_steps = 510000
    args.seed = 0
    return args


def super_res18_s1(args):
    args = super_res18_s0(args)
    args.expId = 'super_s1'
    args.seed = 1
    return args


def super_res18_s2(args):
    args = super_res18_s0(args)
    args.expId = 'super_s2'
    args.seed = 2
    return args
