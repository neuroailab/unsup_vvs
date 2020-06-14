from exp_settings.shared_settings import basic_setting, bs_64_setting


def save_setting(args):
    args.nport = 27007
    args.dbname = 'col'
    args.collname = 'res18'
    return args


def col_setting(args):
    args.init_lr = 0.00001
    args.weight_decay = 1e-4
    args.tpu_flag = 1
    args.pathconfig = 'col_resnet18.cfg'
    args.dataconfig = 'dataset_config_new_colimn.json'
    args.whichopt = 1
    args.adameps = 1e-8
    args.no_prep = 1
    args.with_rep = 1
    return args


def res18_col_s0(args):
    args = basic_setting(args)
    args = save_setting(args)
    args = bs_64_setting(args)
    args = col_setting(args)

    args.expId = 'col_s0'
    args.lr_boundaries = None
    args.train_num_steps = 301 * 10010
    return args


def res18_col_s1(args):
    args = res18_col_s0(args)
    args.expId = 'col_s1'
    args.seed = 1
    return args


def res18_col_s2(args):
    args = res18_col_s0(args)
    args.expId = 'col_s2'
    args.seed = 2
    return args
