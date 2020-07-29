from exp_settings.shared_settings import basic_setting, bs_64_setting


def save_setting(args):
    args.nport = 27007
    args.dbname = 'rp'
    args.collname = 'res18'
    return args


def rp_setting(args):
    args.init_lr = 0.01
    args.weight_decay = 1e-4
    args.tpu_flag = 1
    args.network_func = 'other_tasks.get_rp_resnet_18_bg'
    args.dataconfig = 'dataset_config_new_rpimn.json'
    args.with_rep = 1
    return args


def res18_rp_s0(args):
    args = basic_setting(args)
    args = save_setting(args)
    args = bs_64_setting(args)
    args = rp_setting(args)

    args.lr_boundaries = '779998,1259988'
    args.train_num_steps = 1359988
    args.expId = 'rp_s0'
    return args


def res18_rp_s1(args):
    args = res18_rp_s0(args)
    args.expId = 'rp_s1'
    args.seed = 1
    return args


def res18_rp_s2(args):
    args = res18_rp_s0(args)
    args.expId = 'rp_s2'
    args.seed = 2
    return args
