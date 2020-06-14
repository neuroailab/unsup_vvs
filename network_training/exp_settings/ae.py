from exp_settings.shared_settings import basic_setting, bs_128_less_save_setting


def save_setting(args):
    args.nport = 27007
    args.dbname = 'ae'
    args.collname = 'res18'
    return args


def ae_setting(args):
    args.dataconfig = 'dataset_task_config_image_ae_l1.json'
    args.init_lr = 0.1
    args.with_rep = 1
    args.network_func = 'other_tasks.get_ae_resnet_18'
    args.weight_decay = 1e-4
    return args


def res18_ae_s0(args):
    args = basic_setting(args)
    args = save_setting(args)
    args = bs_128_less_save_setting(args)
    args = ae_setting(args)

    args.expId = 'ae_s0'
    args.lr_boundaries = '660011,1100011'
    args.train_num_steps = 1210000
    return args


def res18_ae_s1(args):
    args = res18_ae_s0(args)
    args.expId = 'ae_s1'
    args.seed = 1
    return args


def res18_ae_s2(args):
    args = res18_ae_s0(args)
    args.expId = 'ae_s2'
    args.seed = 2
    return args
