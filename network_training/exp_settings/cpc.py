from exp_settings.shared_settings import basic_setting, bs_128_less_save_setting


def save_setting(args):
    args.nport = 27007
    args.dbname = 'cpc'
    args.collname = 'res18'
    return args


def cpc_setting(args):
    args.init_lr = 2e-4
    args.whichopt = 1
    args.adameps = 1e-08
    args.dataconfig = 'dataset_task_config_image_cpc.json'
    args.network_func = 'other_tasks.get_cpc_resnet_18'
    return args


def res18_cpc_s0(args):
    args = basic_setting(args)
    args = save_setting(args)
    args = bs_128_less_save_setting(args)
    args = cpc_setting(args)

    args.expId = 'cpc_s0'
    args.train_num_steps = 1310000
    return args


def res18_cpc_s1(args):
    args = res18_cpc_s0(args)
    args.expId = 'cpc_s1'
    args.seed = 1
    return args


def res18_cpc_s2(args):
    args = res18_cpc_s0(args)
    args.expId = 'cpc_s2'
    args.seed = 2
    return args
