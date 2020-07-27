from unsup_vvs.network_training.exp_settings.shared_settings import basic_setting, basic_mean_teacher, \
        bs_128_setting, basic_inst_setting, basic_cate_setting, \
        basic_res18, mt_fst_ramp_down_setting


def cate_res50(args):
    args = basic_res18(args)
    args.nport = 27007
    args.dbname = 'cate_aug'
    args.collname = 'res50'
    args.expId = 'ctl'
    args.resnet_prep = True
    args.resnet_prep_size = True
    args.lr_boundaries = "160000,310000,460000"
    args.train_num_steps = 510000
    args.network_func = 'get_resnet_50'
    return args


def cate_sobel_res18(args):
    args = basic_res18(args)
    args.nport = 27007
    args.dbname = 'cate_aug'
    args.collname = 'res18'
    args.input_mode = 'sobel'
    args.expId = 'sobel'
    args.resnet_prep = True
    args.resnet_prep_size = True
    args.lr_boundaries = "160000,310000,460000"
    args.train_num_steps = 510000
    return args


def cate_res18_inst_prep(args):
    args = basic_res18(args)
    args.nport = 27007
    args.dbname = 'cate_aug'
    args.collname = 'res18'
    args.expId = 'inst_prep'
    args.resnet_prep = False
    args.resnet_prep_size = False
    args.lr_boundaries = "160000,310000,460000"
    args.train_num_steps = 510000
    return args


def cate_res18_exp0(args):
    args = basic_res18(args)
    args.nport = 27007
    args.dbname = 'cate_aug'
    args.collname = 'res18'
    args.expId = 'exp_seed0'
    args.resnet_prep = True
    args.resnet_prep_size = True
    args.lr_boundaries = "160000,310000,460000"
    args.train_num_steps = 510000
    args.seed = 0
    return args


def cate_fc18_exp0(args):
    args = basic_res18(args)
    args.nport = 26001
    args.dbname = 'cate_aug'
    args.collname = 'fc18'
    args.expId = 'exp_seed0'
    args.resnet_prep = True
    args.resnet_prep_size = True
    args.lr_boundaries = "160000,310000,460000"
    args.train_num_steps = 510000
    args.seed = 0
    args.network_func = 'get_fc18'
    return args


def cate_res18_exp0_more_save(args):
    args = cate_res18_exp0(args)
    args.nport = 26001
    args.expId = 'exp_seed0_more_save'
    args.train_num_steps = 5005
    args.fre_filter = 1001
    args.fre_valid = 1001 
    return args


def cate_vgglike_res19_exp0(args):
    args = cate_res18_exp0(args)
    args.nport = 26001
    args.expId = 'exp_vgglike_seed0'
    args.network_func = 'combine_irla_others.get_resnet_vgglike_19'
    args.lr_boundaries = "160000,310000,450000"
    return args


def cate_vgglike_sm_res19_exp0(args):
    args = cate_res18_exp0(args)
    args.nport = 26001
    args.expId = 'exp_vgglike_sm_seed0'
    args.network_func = 'combine_irla_others.get_resnet_vgglike_19'
    args.network_func_kwargs = '{"strides": [2, 1]}'
    return args


def cate_res18_exp0_bn_wd(args):
    args = cate_res18_exp0(args)
    args.expId = 'exp_seed0_bn_wd'
    args.enable_bn_weight_decay = True
    args.global_weight_decay = args.weight_decay
    return args


def cate_res18_exp1(args):
    args = cate_res18_exp0(args)
    args.expId = 'exp_seed1'
    args.seed = 1
    return args


def cate_res18_exp2(args):
    args = cate_res18_exp0(args)
    args.expId = 'exp_seed2'
    args.seed = 2
    return args


def cate_res18_exp3(args):
    args = cate_res18_exp0(args)
    args.expId = 'exp_seed3'
    args.seed = 3
    return args


def part10_cate_res18_fix(args):
    args = basic_res18(args)
    args.dbname = 'combine_instance'
    args.expId = 'part10_cate_res18_fix'
    args.whichimagenet = 'part10'
    return args


def part3_cate_res18(args):
    args = basic_res18(args)
    args.dbname = 'combine_instance'
    args.expId = 'part3_cate_res18'
    args.whichimagenet = 'part3'
    return args


def part5_cate_res18(args):
    args = basic_res18(args)
    args.dbname = 'combine_instance'
    args.expId = 'part5_cate_res18'
    args.whichimagenet = 'part5'
    return args


def mean_teacher_res18(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)

    args.weight_decay = 1e-4
    args.init_lr = 0.03
    args.network_func = 'get_mean_teacher_resnet_18'

    args.dbname = 'combine_instance'
    args.expId = 'res18_mt_0'
    return args


def mt_part10_res18_fx(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_fst_ramp_down_setting(args)

    args.whichimagenet = 'part10'
    args.expId = 'mt_part10_res18_fx'
    return args


def mt_part10_res50(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_fst_ramp_down_setting(args)

    args.whichimagenet = 'part10'
    args.expId = 'mt_part10_res50_fx'
    args.nport = 27006
    args.network_func = 'get_mean_teacher_resnet_50'
    args.mt_ramp_down_epoch = 120
    args.fre_filter = 100100
    args.fre_cache_filter = 10010
    return args


def mt_part5_res18(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_fst_ramp_down_setting(args)

    args.whichimagenet = 'part5'
    args.expId = 'mt_part5_res18'
    return args


def mt_part3_res18(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_fst_ramp_down_setting(args)

    args.whichimagenet = 'part3'
    args.expId = 'mt_part3_res18'
    return args

def mt_load_from_inst(args):
    args.mt_ckpt_load_dict = 1
    args.ema_zerodb = 1
    args.ckpt_file = \
            '/mnt/fs3/chengxuz/visualmaster_related/instance_ckpt/'\
            + 'checkpoint-2560000'
    return args


def mt_part3_res18_inst(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_fst_ramp_down_setting(args)
    args = mt_load_from_inst(args)

    args.whichimagenet = 'part3'
    args.expId = 'mt_part3_res18_inst'
    return args


def mt_part5_res18_inst(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_fst_ramp_down_setting(args)
    args = mt_load_from_inst(args)

    args.whichimagenet = 'part5'
    args.expId = 'mt_part5_res18_inst'
    return args


def mt_part10_res18_inst(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = mt_fst_ramp_down_setting(args)
    args = mt_load_from_inst(args)

    args.whichimagenet = 'part10'
    args.expId = 'mt_part10_res18_inst'
    return args


def inst_mt_res18_fst_p3_from_inst_fx(args):
    args = inst_mt_res18_faster(args)
    args = mt_load_from_inst(args)
    args.whichimagenet = 'part3'
    args.expId = 'inst_mt_res18_fst_p3_from_inst_fx'
    return args


def inst_mt_res18_fst_p5_from_inst_fx(args):
    args = inst_mt_res18_faster(args)
    args = mt_load_from_inst(args)
    args.whichimagenet = 'part5'
    args.expId = 'inst_mt_res18_fst_p5_from_inst_fx'
    return args


def inst_mt_res18_faster(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = basic_inst_setting(args)
    args = mt_fst_ramp_down_setting(args)
    args.network_func = 'get_mean_teacher_and_inst_resnet_18'
    return args


def inst_mt_res18_fst_p3(args):
    args = inst_mt_res18_faster(args)
    args.whichimagenet = 'part3'
    args.expId = 'inst_mt_res18_fst_p3'
    return args


def inst_mt_res18_fst_p5(args):
    args = inst_mt_res18_faster(args)
    args.whichimagenet = 'part5'
    args.expId = 'inst_mt_res18_fst_p5'
    return args


def inst_and_mean_teacher_res18(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = basic_inst_setting(args)

    args.dbname = 'combine_instance'
    args.expId = 'res18_inst_mt_cont'
    args.network_func = 'get_mean_teacher_and_inst_resnet_18'

    args.lr_boundaries = "480000"
    args.loadexpId = 'res18_inst_mt_1'
    args.loadstep = 480480
    return args


def inst_and_mt_rmp_dwn_res18(args):
    args = basic_setting(args)
    args = basic_mean_teacher(args)
    args = bs_128_setting(args)
    args = basic_inst_setting(args)

    args.dbname = 'combine_instance'
    args.expId = 'inst_and_mt_rmp_dwn_res18'
    args.network_func = 'get_mean_teacher_and_inst_resnet_18'
    args.mt_ramp_down = 1
    args.mt_ramp_down_epoch = 110
    args.target_lr = 0.0125
    return args


def inst_and_mt_rmp_dwn_lg_res18(args):
    args = inst_and_mt_rmp_dwn_res18(args)
    args.mt_ramp_down_epoch = 330
    args.expId = 'inst_and_mt_rmp_dwn_lg_res18'
    return args


def inst_mt_rmp_dwn_lg_sm_res18(args):
    args = inst_and_mt_rmp_dwn_res18(args)
    args.mt_ramp_down_epoch = 330
    args.target_lr = 0.00375
    args.expId = 'inst_mt_rmp_dwn_lg_sm_res18'
    return args


def inst_mt_lg_sm_p10_res18_fx(args):
    args = inst_and_mt_rmp_dwn_res18(args)
    args.mt_ramp_down_epoch = 330
    args.target_lr = 0.00375
    args.expId = 'inst_mt_lg_sm_p10_res18_fx'
    args.whichimagenet = 'part10'
    return args


def inst_mt_lg_sm_p5_res18(args):
    args = inst_and_mt_rmp_dwn_res18(args)
    args.mt_ramp_down_epoch = 330
    args.target_lr = 0.00375
    args.expId = 'inst_mt_lg_sm_p5_res18'
    args.whichimagenet = 'part5'
    return args


def inst_and_cate_res18(args):
    args = basic_setting(args)
    args.init_type = 'variance_scaling_initializer'
    args.tpu_flag = 1
    args.weight_decay = 1e-4
    args.imgnt_w_idx = True
    args.whichopt = 0
    args.dataconfig = 'dataset_task_config_image_ir_cate.json'

    args.init_lr = 0.03
    args.lr_boundaries = "500000,800000,960000"

    args.network_func = 'get_resnet_18_inst_and_cate'

    args.valinum = 250
    args.valbatchsize = 200
    args.whichimagenet = 'full_widx'
    args.batchsize = 256

    args.dbname = 'combine_instance'
    args.expId = 'res18_inst_cate_1'
    args.fre_filter = 10010
    args.fre_valid = 5005
    return args


def inst_and_cate_res18_early(args):
    args = inst_and_cate_res18(args)
    args.network_func = 'get_resnet_18_inst_and_cate_early_memory'
    args.expId = 'res18_inst_cate_early'
    return args


def inst_and_cate_res18_even_early(args):
    args = inst_and_cate_res18(args)
    args.network_func = 'get_resnet_18_inst_and_cate_even_early_memory'
    args.expId = 'res18_inst_cate_even_early'
    return args


def inst_and_cate_res18_early_enc6(args):
    args = inst_and_cate_res18(args)
    args.network_func = 'get_resnet_18_inst_and_cate_memory_enc6'
    args.expId = 'res18_inst_cate_early_enc6'
    args.train_num_steps = 50050
    return args


def inst_cate_res18_sep_p3(args):
    args = inst_and_cate_res18(args)

    args.inst_cate_sep = True
    args.lr_boundaries = None
    args.dataconfig = 'dataset_config_image_wun.cfg'
    args.expId = 'inst_cate_res18_sep_p3_fx'
    args.network_func = 'get_resnet_18_inst_and_cate_sep'
    args.whichimagenet = 'part3'

    args.fre_filter = 100100
    args.fre_cache_filter = 10010

    args.lr_boundaries = "270000,450000"
    return args


def inst_cate_res18_sep_p5(args):
    args = inst_cate_res18_sep_p3(args)

    args.expId = 'inst_cate_res18_sep_p5_fx'
    args.whichimagenet = 'part5'
    args.lr_boundaries = "200000,380000"
    return args


def basic_instance_setting(args):
    args.init_type = 'instance_resnet'
    args.tpu_flag = 1
    args.weight_decay = 1e-4
    args.instance_task = True
    args.whichopt = 0
    args.dataconfig = 'dataset_config_image.cfg'
    if args.init_lr==0.01:
        # TODO: fix this so that 0.01 is possible
        args.init_lr = 0.03
    return args


def imagenet_instance_setting(args):
    args = basic_setting(args)
    args = basic_instance_setting(args)
    
    args.valinum = 500
    args.valbatchsize = 100
    args.whichimagenet = 'full_widx'
    args.network_func = 'get_resnet_18'
    args.network_func_kwargs = '{"num_cat": 128}'
    return args


def imagenet_instance_setting_res101_3blk(args):
    args = basic_setting(args)
    args = basic_instance_setting(args)
    
    args.valinum = 250
    args.valbatchsize = 200
    args.whichimagenet = 'full_widx'
    args.network_func = 'get_resnet_101_3blk'
    args.network_func_kwargs = '{"num_cat": 128}'
    return args


def infant_instance_setting(args):
    args = basic_setting(args)
    args = basic_instance_setting(args)
    
    args.valinum = 123
    args.valbatchsize = 100
    args.whichimagenet = 'full_new_widx'
    args.instance_data_len = 305228
    args.network_func = 'get_resnet_18'
    args.network_func_kwargs = '{"num_cat": 128}'
    return args
