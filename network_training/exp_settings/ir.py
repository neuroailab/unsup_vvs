import exp_settings.shared_settings as shared_sts


def save_setting(args):
    args.nport = 27007
    args.dbname = 'pub_ir'
    args.collname = 'res18'
    return args


def basic_inst_setting(args):
    args = shared_sts.basic_inst_setting(args)
    args.tpu_flag = 1
    args.whichopt = 0
    args.whichimagenet = 'full_widx'
    args.init_type = 'variance_scaling_initializer'
    return args


def res18_ir_s0(args):
    args = shared_sts.basic_setting(args)
    args = basic_inst_setting(args)

    args = save_setting(args)
    args = shared_sts.bs_128_less_save_setting(args)
    args.network_func = 'get_resnet_18'
    args.network_func_kwargs = '{"num_cat": 128}'
    args.dataconfig = 'dataset_task_config_image_ir.json'
    args.expId = 'ir_s0'
    args.lr_boundaries = '1780011,2400011'
    args.train_num_steps = 2610011
    return args


def res18_ir_s1(args):
    args = res18_ir(args)
    args.expId = 'ir_s1'
    args.seed = 1
    return args


def res18_ir_s2(args):
    args = res18_ir(args)
    args.expId = 'ir_s2'
    args.seed = 2
    return args
