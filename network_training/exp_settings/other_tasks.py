from exp_settings.shared_settings import basic_setting, bs_64_setting, \
        bs_128_setting, basic_inst_setting, basic_cate_setting, basic_res18, \
        bs_128_less_save_setting


def save_setting(args):
    args.nport = 27007
    args.dbname = 'other_tasks'
    args.collname = 'res18'
    return args


def rp_setting(args):
    args.init_lr = 0.01
    args.weight_decay = 1e-4
    args.tpu_flag = 1
    args.network_func = 'other_tasks.get_rp_resnet_18'
    args.dataconfig = 'dataset_config_new_rpimn.json'
    args.with_rep = 1
    return args


def res18_rp_imagenet(args):
    args = basic_setting(args)
    args = save_setting(args)
    args = bs_64_setting(args)
    args = rp_setting(args)

    args.expId = 'rp_imgnt_fx'
    args.lr_boundaries = '779998,1259988'
    return args


def res18_rp_imagenet_clr(args):
    args = res18_rp_imagenet(args)
    args.expId = 'rp_imgnt_clr_fx'
    args.dataconfig = 'dataset_config_new_rpimn_clr.json'
    return args


def res18_rp_imagenet_bg(args):
    args = res18_rp_imagenet(args)
    args.expId = 'rp_imgnt_bg'
    args.network_func = 'other_tasks.get_rp_resnet_18_bg'
    return args


def res18_rp_imagenet_tpu(args):
    args = res18_rp_imagenet(args)
    args.expId = 'rp_imgnt_tpu'
    args.network_func = 'other_tasks.get_rp_resnet_18_bg'
    args.color_norm = 2 # Just divide by 255
    args.rp_dp_tl = 1
    return args


def res18_rp_tpu_test(args):
    args = res18_rp_imagenet(args)
    args.network_func = 'other_tasks.get_rp_resnet_18_bg'
    args.expId = 'res18_rp_tpu_test'
    args.no_prep = 1
    args.rp_dp_tl = 1
    step_num = 1181162
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/rp_res18/model.ckpt-%i' % step_num
    args.add_batchname = '_rp'
    args.fre_valid = 1181166
    args.valinum = 1000
    args.valbatchsize = 50
    return args


def dp_pbr_setting(args):
    args.init_lr = 0.1
    args.whichopt = 1
    args.weight_decay = 1e-4
    args.tpu_flag = 1
    args.pathconfig = 'depth_resnet18_up4.cfg'
    args.dataconfig = 'dataset_config_pbr.cfg'
    return args


def res18_dp_pbr(args):
    args = basic_setting(args)
    args = save_setting(args)
    args = bs_64_setting(args)
    args = dp_pbr_setting(args)

    args.expId = 'dp_pbr'
    args.lr_boundaries = '639998,1000011,1539988'
    return args


def res18_dp_ps(args):
    args = res18_dp_pbr(args)
    args.dataconfig = 'dataset_config_scenepbr.cfg'
    args.expId = 'dp_ps'
    args.lr_boundaries = '579987,1039988,1299988'
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


def res18_col_imagenet(args):
    args = basic_setting(args)
    args = save_setting(args)
    args = bs_64_setting(args)
    args = col_setting(args)

    args.expId = 'col_imgnt'
    args.lr_boundaries = None
    return args


def res18_col_imagenet_fst(args):
    args = res18_col_imagenet(args)
    args.expId = 'col_imgnt_fst'
    args.init_lr = 0.01
    return args


def ae_setting(args):
    args.dataconfig = 'dataset_task_config_image_ae.json'
    args.init_lr = 0.1
    args.network_func = 'other_tasks.get_ae_resnet_18'
    return args


def res18_AE_imagenet(args):
    args = basic_setting(args)
    args = save_setting(args)
    args = bs_128_less_save_setting(args)
    args = ae_setting(args)

    args.expId = 'ae'
    args.lr_boundaries = '660011,1100011'
    return args


def res18_AE_imagenet_l1(args):
    args = res18_AE_imagenet(args)
    args.nport = 26001
    args.expId = 'ae_l1_2'
    args.dataconfig = 'dataset_task_config_image_ae_l1.json'
    args.with_rep = 1
    args.weight_decay = 1e-4
    args.lr_boundaries = '350011,710011'
    return args


def res18_AE_imagenet_wd_l2(args):
    args = res18_AE_imagenet(args)
    args.nport = 26001
    args.expId = 'ae_wd_l2'
    args.with_rep = 1
    args.weight_decay = 1e-4
    args.lr_boundaries = '410011,570011'
    return args


def load_from_save_setting(args):
    args.loadport = 27007
    args.load_dbname = 'other_tasks'
    args.load_collname = 'res18'
    return args


def res18_AE_imagenet_visualize(args):
    args = basic_setting(args)
    args = save_setting(args)
    args = load_from_save_setting(args)
    args = bs_128_less_save_setting(args)
    args = ae_setting(args)

    args.loadexpId = 'ae'
    args.expId = 'ae_vis'
    args.with_feat = 1
    args.valinum = 10
    return args


def res18_AE_imagenet_seed1(args):
    args = res18_AE_imagenet(args)
    args.expId = 'ae_seed1'
    args.seed = 1
    return args


def res18_AE_imagenet_seed2(args):
    args = res18_AE_imagenet(args)
    args.expId = 'ae_seed2'
    args.seed = 2
    return args


def cpc_setting(args):
    args.init_lr = 2e-4
    args.whichopt = 1
    args.adameps = 1e-08
    args.dataconfig = 'dataset_task_config_image_cpc.json'
    args.network_func = 'other_tasks.get_cpc_resnet_18'
    return args


def res18_cpc_imagenet(args):
    args = basic_setting(args)
    args = save_setting(args)
    args = bs_128_less_save_setting(args)
    args = cpc_setting(args)

    args.expId = 'cpc'
    return args


def res18_cpc_imagenet_tpu(args):
    args = basic_setting(args)
    args = bs_128_setting(args)
    args = cpc_setting(args)
    args.nport = 27001

    args.expId = 'cpc'
    args.tpu_task = 'cpc'
    args.cacheDirPrefix = 'gs://cx_visualmaster/'
    args.valinum = 390
    args.valbatchsize = 128
    return args


def res18_cpc_imagenet_tpu_seed1(args):
    args = res18_cpc_imagenet_tpu(args)
    args.expId = 'cpc_seed1'
    args.seed = 1
    return args


def res18_cpc_imagenet_tpu_seed2(args):
    args = res18_cpc_imagenet_tpu(args)
    args.expId = 'cpc_seed2'
    args.seed = 2
    return args
