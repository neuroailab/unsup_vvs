def cate_seed0(args):
    args.load_step = 505505
    args.setting_name = 'cate_res18_exp0'
    return args


def cate_seed0_ckpt(args):
    args.setting_name = 'cate_res18_exp0'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/brainscore_model_caches/cate_aug/res18/exp_seed0/checkpoint-505505'
    args.identifier = 'cate-seed0-ckpt'
    return args


def cate_seed1(args):
    args.load_step = 505505
    args.setting_name = 'cate_res18_exp1'
    return args


def cate_seed2(args):
    args.load_step = 505505
    args.setting_name = 'cate_res18_exp2'
    return args


def cate_input(args):
    args.load_step = 505505
    args.setting_name = 'cate_res18_exp0'
    args.just_input = True
    return args


def cate_vgglike_seed0(args):
    args.load_step = 505505
    args.setting_name = 'cate_vgglike_res19_exp0'
    all_layers = ['encode_1'] + ['encode_2.conv'] + ['encode_%i' % i for i in range(2, 11)]
    args.layers = ','.join(all_layers)
    return args


def cate_vgglike_sm_seed0(args):
    args.load_step = 505505
    args.setting_name = 'cate_vgglike_sm_res19_exp0'
    all_layers = ['encode_1'] + ['encode_2.conv'] + ['encode_%i' % i for i in range(2, 11)]
    args.layers = ','.join(all_layers)
    return args


def cate_seed0_ep02(args):
    args.load_step = 1001 * 1
    args.setting_name = 'cate_res18_exp0_more_save'
    return args


def cate_seed0_ep04(args):
    args.load_step = 1001 * 2
    args.setting_name = 'cate_res18_exp0_more_save'
    return args


def cate_seed0_ep06(args):
    args.load_step = 1001 * 3
    args.setting_name = 'cate_res18_exp0_more_save'
    return args


def cate_seed0_ep08(args):
    args.load_step = 1001 * 4
    args.setting_name = 'cate_res18_exp0_more_save'
    return args


def cate_seed0_ep1(args):
    args.load_step = 5005 * 1
    args.setting_name = 'cate_res18_exp0'
    return args


def cate_seed0_ep2(args):
    args.load_step = 5005 * 2
    args.setting_name = 'cate_res18_exp0'
    return args


def cate_seed0_ep3(args):
    args.load_step = 5005 * 3
    args.setting_name = 'cate_res18_exp0'
    return args


def cate_seed0_ep4(args):
    args.load_step = 5005 * 4
    args.setting_name = 'cate_res18_exp0'
    return args


def cate_seed0_ep5(args):
    args.load_step = 5005 * 5
    args.setting_name = 'cate_res18_exp0'
    return args


def cate_seed0_ep10(args):
    args.load_step = 5005 * 10
    args.setting_name = 'cate_res18_exp0'
    return args


def cate_seed0_ep20(args):
    args.load_step = 5005 * 20
    args.setting_name = 'cate_res18_exp0'
    return args


def cate_seed0_ep35(args):
    args.load_step = 5005 * 35
    args.setting_name = 'cate_res18_exp0'
    return args


def cate_seed0_ep55(args):
    args.load_step = 5005 * 55
    args.setting_name = 'cate_res18_exp0'
    return args


def cate_seed0_ep65(args):
    args.load_step = 5005 * 65
    args.setting_name = 'cate_res18_exp0'
    return args


def cate_seed0_ep85(args):
    args.load_step = 5005 * 85
    args.setting_name = 'cate_res18_exp0'
    return args


def cate_p03(args):
    args.load_step = 390390
    args.setting_name = 'part3_cate_res18'
    return args


def cate_res50(args):
    args.load_step = 505505
    args.setting_name = 'cate_res50'
    args.layers = ','.join(['encode_%i' % i for i in range(1, 18)]) # no first conv layer
    return args
