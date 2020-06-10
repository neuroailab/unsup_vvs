from nf_exp_settings.shared_settings import bs_basic_setting, bs_res18_setting


def hvm_v4_setting(args):
    args.benchmark = 'dicarlo.Majaj2015.V4-mask'
    return args


def hvm_v4_pls_setting(args):
    args.benchmark = 'dicarlo.Majaj2015.V4-pls'
    return args


def hvm_it_setting(args):
    args.benchmark = 'dicarlo.Majaj2015.IT-mask'
    return args


def hvm_it_pls_setting(args):
    args.benchmark = 'dicarlo.Majaj2015.IT-pls'
    return args


def cate_res18_hvm_v4(args):
    args = bs_basic_setting(args)
    args = bs_res18_setting(args)
    args = hvm_v4_setting(args)

    step_num = 64 * 10009
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/res18/model.ckpt-%i' % step_num
    #args.expId = 'cate_res18_hvm_v4'
    args.expId = 'cate_res18_hvm_v4_new'
    return args


def cate_res18_hvm_it(args):
    args = bs_basic_setting(args)
    args = bs_res18_setting(args)
    args = hvm_it_setting(args)

    step_num = 64 * 10009
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/res18/model.ckpt-%i' % step_num
    #args.expId = 'cate_res18_hvm_it'
    args.expId = 'cate_res18_hvm_it_new'
    return args


def cate_res18_hvm_v4_pls(args):
    args = cate_res18_hvm_v4(args)
    args = hvm_v4_pls_setting(args)
    args.expId = 'cate_res18_hvm_v4_pls'
    return args


def cate_res18_hvm_it_pls(args):
    args = cate_res18_hvm_it(args)
    args = hvm_it_pls_setting(args)
    args.expId = 'cate_res18_hvm_it_pls'
    return args


def cate_seed0_res18_hvm_v4_pls(args):
    args = bs_basic_setting(args)
    args = bs_res18_setting(args)
    args = hvm_v4_pls_setting(args)

    args.expId = 'cate_seed0_res18_hvm_v4_pls'
    args.setting_name = 'cate_res18_exp0'
    args.load_train_setting_func = 'cate_res18_exp0'
    args.loadstep = 490490
    return args


def cate_seed0_res18_hvm_it_pls(args):
    args = bs_basic_setting(args)
    args = bs_res18_setting(args)
    args = hvm_it_pls_setting(args)

    args.expId = 'cate_seed0_res18_hvm_it_pls'
    args.setting_name = 'cate_res18_exp0'
    args.load_train_setting_func = 'cate_res18_exp0'
    args.loadstep = 490490
    return args


def cate_seed1_res18_hvm_v4_pls(args):
    args = cate_seed0_res18_hvm_v4_pls(args)
    args.expId = 'cate_seed1_res18_hvm_v4_pls'
    args.setting_name = 'cate_res18_exp1'
    args.load_train_setting_func = 'cate_res18_exp1'
    return args


def cate_seed1_res18_hvm_it_pls(args):
    args = cate_seed0_res18_hvm_it_pls(args)
    args.expId = 'cate_seed1_res18_hvm_it_pls'
    args.setting_name = 'cate_res18_exp1'
    args.load_train_setting_func = 'cate_res18_exp1'
    return args
