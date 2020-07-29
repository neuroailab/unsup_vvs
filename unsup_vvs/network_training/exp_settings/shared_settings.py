def basic_setting(args):
    # General setting useful for all tasks
    args.use_faster_mgpu = True
    args.use_dataset_inter = True
    args.with_feat = 0
    args.color_norm = 1
    args.withclip = 0
    args.nport = 27009
    return args

def basic_mean_teacher(args):
    args.init_type = 'variance_scaling_initializer'
    args.tpu_flag = 1
    args.mean_teacher = True
    args.cons_ramp_len = 100100
    args.whichopt = 3
    args.dataconfig = 'dataset_config_image_wun.cfg'
    args.whichimagenet = 'part3'
    return args


def bs_128_setting(args):
    args.valinum = 500
    args.valbatchsize = 100
    args.batchsize = 128
    args.fre_filter = 10010
    args.fre_valid = 10010 
    return args


def bs_128_less_save_setting(args):
    args = bs_128_setting(args)
    args.fre_filter = 100100
    args.fre_cache_filter = 10010
    return args


def bs_64_setting(args):
    args.valinum = 500
    args.valbatchsize = 100
    args.batchsize = 64
    args.fre_filter = 100100
    args.fre_cache_filter = 20020
    args.fre_valid = 20020 
    return args


def bs_64_less_save_setting(args):
    args = bs_64_setting(args)
    args.fre_filter = 100100
    args.fre_cache_filter = 10010
    return args


def bs_256_setting(args):
    args.valinum = 250
    args.valbatchsize = 200
    args.batchsize = 256
    args.fre_filter = 5005
    args.fre_valid = 5005 
    return args


def bs_256_less_save_setting(args):
    args = bs_256_setting(args)
    args.fre_filter = 50050
    args.fre_cache_filter = 5005
    return args


def basic_inst_setting(args):
    args.weight_decay = 1e-4
    args.instance_task = True
    args.init_lr = 0.03
    return args


def basic_cate_setting(args):
    args.dataconfig = 'dataset_config_image.cfg'
    args.init_type = 'variance_scaling_initializer'
    args.tpu_flag = 1
    args.weight_decay = 1e-4
    args.init_lr = 0.1
    return args


def basic_res18(args):
    args = basic_setting(args)
    args = bs_256_setting(args)
    args = basic_cate_setting(args)
    args.network_func = 'get_resnet_18'
    args.lr_boundaries = "160000,310000"
    return args


def mt_fst_ramp_down_setting(args):
    args.dbname = 'combine_instance'
    args.weight_decay = 1e-4
    args.init_lr = 0.00375
    args.network_func = 'get_mean_teacher_resnet_18'
    args.mt_ramp_down = 1
    args.mt_ramp_down_epoch = 330
    return args
