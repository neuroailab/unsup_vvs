def tpu_basic_setting(args):
    # General setting useful for all tasks
    args.with_feat = 0
    args.color_norm = 1
    args.withclip = 0
    args.nport = 27001
    args.init_type = 'variance_scaling_initializer'
    args.cacheDirPrefix = 'gs://cx_visualmaster/'
    args.tpu_flag = 1
    return args


def bs256_setting(args):
    args.batchsize = 256
    args.valinum = 196
    args.fre_filter = 10009
    args.fre_valid = 10009
    return args


def tpu_full_cate_setting(args):
    args = bs256_setting(args)
    
    args.tpu_task = 'imagenet'
    args.whichimagenet = 'full'
    args.whichopt = 0
    args.global_weight_decay = 1e-4
    args.resnet_prep = True
    args.dataconfig = 'dataset_config_image.cfg'
    args.lr_boundaries = '150150,300300,450450'
    return args


def tpu_vgg16(args):
    args = tpu_basic_setting(args)
    args = tpu_full_cate_setting(args)

    args.namefunc = 'tpu_combine_tfutils_general'
    args.network_func = 'get_vgg16'

    args.expId = 'tpu_vgg16'
    return args


def tpu_p10_mt_setting(args):
    args = bs256_setting(args)

    args.tpu_task = 'mean_teacher'
    args.whichimagenet = 'part10'
    args.whichopt = 3
    args.global_weight_decay = 5e-5
    args.resnet_prep = True
    args.dataconfig = 'dataset_config_image_wun.cfg'
    args.mt_ramp_down = 1
    args.cons_ramp_len = 50000
    args.init_lr = 0.1
    args.mean_teacher = 1
    return args


def tpu_mt_res50(args):
    args = tpu_basic_setting(args)
    args = tpu_p10_mt_setting(args)

    args.namefunc = 'tpu_combine_tfutils_general'
    args.network_func = 'get_mean_teacher_resnet_50'

    args.expId = 'tpu_mt_res50'
    return args
