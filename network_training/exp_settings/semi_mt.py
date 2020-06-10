from exp_settings.shared_settings import basic_setting, basic_mean_teacher, bs_128_setting, \
        mt_fst_ramp_down_setting


def mt_semi_basic(args):
    args.nport = 27006
    args.mt_ramp_down_epoch = 120
    args.fre_filter = 100100
    args.fre_cache_filter = 10010
    args.dbname = 'combine_instance'
    args.weight_decay = 1e-4
    args.init_lr = 0.00375
    args.mt_ramp_down = 1
    return args


def mt_part10_res50(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_semi_basic(args)

    args.whichimagenet = 'part10'
    args.expId = 'mt_part10_res50_fx'
    args.network_func = 'get_mean_teacher_resnet_50'
    return args


def mt_part30_part10_res18(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_semi_basic(args)

    args.whichimagenet = 'part10'
    args.expId = 'mt_part30_part10_res18'
    args.network_func = 'get_mean_teacher_resnet_18'
    args.dataconfig = 'dataset_config_image_wun_p30.cfg'
    return args


def mt_part70_part10_res18(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_semi_basic(args)

    args.whichimagenet = 'part10'
    args.expId = 'mt_part70_part10_res18'
    args.network_func = 'get_mean_teacher_resnet_18'
    args.dataconfig = 'dataset_config_image_wun_p70.cfg'
    return args


def mt_part10_part10_res18(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_semi_basic(args)

    args.whichimagenet = 'part10'
    args.expId = 'mt_part10_part10_res18'
    args.network_func = 'get_mean_teacher_resnet_18'
    args.dataconfig = 'dataset_config_image_wun_p10.cfg'
    return args


def mt_part10_res18(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_semi_basic(args)

    args.whichimagenet = 'part10'
    args.expId = 'mt_part10_res18'
    args.network_func = 'get_mean_teacher_resnet_18'
    args.dataconfig = 'dataset_config_image_wun_widx.cfg'
    return args


def mt_part10_res18_relearn(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_semi_basic(args)

    args.whichimagenet = 'part10'
    args.expId = 'mt_part10_res18_relearn'
    args.network_func = 'get_mean_teacher_resnet_18'
    args.dataconfig = 'dataset_config_image_wun_widx.cfg'

    args.loadexpId = 'mt_part10_res18'
    args.drop_global_step = True
    args.loadstep = 1001000
    return args


def mt_part01_res18(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_semi_basic(args)

    args.whichimagenet = 'part1_balanced'
    args.expId = 'mt_part01_res18'
    args.network_func = 'get_mean_teacher_resnet_18'
    args.dataconfig = 'dataset_config_image_wun_widx.cfg'
    return args


def mt_part01_res50(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_semi_basic(args)

    args.whichimagenet = 'part1_balanced'
    args.expId = 'mt_part01_res50'
    args.network_func = 'get_mean_teacher_resnet_50'
    return args


def mt_part3_res18_rep1(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_fst_ramp_down_setting(args)

    args.whichimagenet = 'part3'
    args.expId = 'mt_part3_res18_rep1'
    args.nport = 27007
    return args


def new_save_setting(args):
    args.nport = 26001
    args.dbname = 'mt_frmLA'
    args.collname = 'res18'
    return args


def load_from_LA(args):
    args.mt_ckpt_load_dict = 1
    args.ema_zerodb = 1
    args.ckpt_file = \
            '/mnt/fs4/chengxuz/brainscore_model_caches/irla_and_others/res18/la_s1/'\
            + 'checkpoint-2502500'
    return args


def mt_p01_res18_frmLA(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_semi_basic(args)
    args = new_save_setting(args)
    args = load_from_LA(args)

    args.whichimagenet = 'part1_balanced'
    args.expId = 'mt_p01_res18_frmLA'
    args.network_func = 'get_mean_teacher_resnet_18'
    args.dataconfig = 'dataset_config_image_wun_widx.cfg'
    return args


def mt_p03_res18_frmLA(args):
    args = mt_p01_res18_frmLA(args)
    args.whichimagenet = 'part3'
    args.expId = 'mt_p03_res18_frmLA'
    return args


def mt_p50_res18(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_semi_basic(args)
    args = new_save_setting(args)

    args.whichimagenet = 'part50_balanced'
    args.expId = 'mt_p50_res18'
    args.network_func = 'get_mean_teacher_resnet_18'
    args.dataconfig = 'dataset_config_image_wun_widx.cfg'
    return args


def mt_p50_res18_s1(args):
    args = mt_p50_res18(args)
    args.expId = 'mt_p50_res18_s1'
    args.seed = 1
    return args


def mt_p50_res18_s2(args):
    args = mt_p50_res18(args)
    args.expId = 'mt_p50_res18_s2'
    args.seed = 2
    return args


def mt_p01_res18_s1(args):
    args = mt_p50_res18(args)
    args.whichimagenet = 'part1_balanced'
    args.expId = 'mt_p01_res18_s1'
    args.seed = 1
    return args


def mt_p01_res18_s2(args):
    args = mt_p01_res18_s1(args)
    args.expId = 'mt_p01_res18_s2'
    args.seed = 2
    return args


def mt_p02_res18(args):
    args = mt_p50_res18(args)
    args.whichimagenet = 'part2_balanced'
    args.expId = 'mt_p02_res18'
    return args


def mt_p02_res18_s1(args):
    args = mt_p02_res18(args)
    args.expId = 'mt_p02_res18_s1'
    args.seed = 1
    return args


def mt_p02_res18_s2(args):
    args = mt_p02_res18(args)
    args.expId = 'mt_p02_res18_s2'
    args.seed = 2
    return args


def mt_p03_res18_s1(args):
    args = mt_p50_res18(args)
    args.whichimagenet = 'part3'
    args.expId = 'mt_p03_res18_s1'
    args.seed = 1
    return args


def mt_p03_res18_s2(args):
    args = mt_p03_res18_s1(args)
    args.expId = 'mt_p03_res18_s2'
    args.seed = 2
    return args


def mt_p04_res18(args):
    args = mt_p50_res18(args)
    args.whichimagenet = 'part4_balanced'
    args.expId = 'mt_p04_res18'
    return args


def mt_p04_res18_s1(args):
    args = mt_p04_res18(args)
    args.expId = 'mt_p04_res18_s1'
    args.seed = 1
    return args


def mt_p04_res18_s2(args):
    args = mt_p04_res18(args)
    args.expId = 'mt_p04_res18_s2'
    args.seed = 2
    return args


def mt_p05_res18_s1(args):
    args = mt_p50_res18(args)
    args.whichimagenet = 'part5'
    args.expId = 'mt_p05_res18_s1'
    args.seed = 1
    return args


def mt_p05_res18_s2(args):
    args = mt_p05_res18_s1(args)
    args.expId = 'mt_p05_res18_s2'
    args.seed = 2
    return args


def mt_p06_res18(args):
    args = mt_p50_res18(args)
    args.whichimagenet = 'part6_balanced'
    args.expId = 'mt_p06_res18'
    return args


def mt_p06_res18_s1(args):
    args = mt_p06_res18(args)
    args.expId = 'mt_p06_res18_s1'
    args.seed = 1
    return args


def mt_p06_res18_s2(args):
    args = mt_p06_res18(args)
    args.expId = 'mt_p06_res18_s2'
    args.seed = 2
    return args


def mt_p10_res18_s1(args):
    args = mt_p50_res18(args)
    args.whichimagenet = 'part10'
    args.expId = 'mt_p10_res18_s1'
    args.seed = 1
    return args


def mt_p10_res18_s2(args):
    args = mt_p10_res18_s1(args)
    args.expId = 'mt_p10_res18_s2'
    args.seed = 2
    return args


def mt_p20_res18(args):
    args = mt_p50_res18(args)
    args.whichimagenet = 'part20_balanced'
    args.expId = 'mt_p20_res18'
    return args


def mt_p20_res18_s1(args):
    args = mt_p20_res18(args)
    args.expId = 'mt_p20_res18_s1'
    args.seed = 1
    return args


def mt_p20_res18_s2(args):
    args = mt_p20_res18(args)
    args.expId = 'mt_p20_res18_s2'
    args.seed = 2
    return args
