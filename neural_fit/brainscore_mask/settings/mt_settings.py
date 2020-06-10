def mt_p03_s0(args):
    args.layers = ','.join(
            ['encode_%i' % i for i in range(1, 10)] \
            + ['ema_encode_%i' % i for i in range(1, 10)])
    args.model_type = 'mt_vm_model'
    args.load_step = 1161160
    args.setting_name = 'mt_part3_res18'
    return args


def mt_p03_frm_LA(args):
    args.layers = ','.join(
            ['encode_%i' % i for i in range(1, 10)])
    args.model_type = 'mt_vm_model'
    args.load_step = 100 * 10010
    args.setting_name = 'semi_mt.mt_p03_res18_frmLA'
    return args


def mt_p01_s0(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 10 * 10010
    args.setting_name = 'semi_mt.mt_part01_res18'
    return args


def mt_p01_s1(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 10 * 10010
    args.setting_name = 'semi_mt.mt_p01_res18_s2'
    return args


def mt_p01_s2(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 10 * 10010
    args.setting_name = 'semi_mt.mt_p01_res18_s1'
    return args


def mt_p02_s0(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 10 * 10010
    args.setting_name = 'semi_mt.mt_p02_res18'
    return args


def mt_p02_s1(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 10 * 10010
    args.setting_name = 'semi_mt.mt_p02_res18_s1'
    return args


def mt_p02_s2(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 10 * 10010
    args.setting_name = 'semi_mt.mt_p02_res18_s2'
    return args


def mt_p03_s1(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 30 * 10010
    args.setting_name = 'semi_mt.mt_p03_res18_s1'
    return args


def mt_p03_s2(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 30 * 10010
    args.setting_name = 'semi_mt.mt_p03_res18_s2'
    return args


def mt_p04_s0(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 70 * 10010
    args.setting_name = 'semi_mt.mt_p04_res18'
    return args


def mt_p04_s1(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 70 * 10010
    args.setting_name = 'semi_mt.mt_p04_res18_s1'
    return args


def mt_p04_s2(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 70 * 10010
    args.setting_name = 'semi_mt.mt_p04_res18_s2'
    return args


def mt_p05_s0(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 140 * 10010
    args.setting_name = 'mt_part5_res18'
    return args


def mt_p05_s1(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 90 * 10010
    args.setting_name = 'semi_mt.mt_p05_res18_s1'
    return args


def mt_p05_s2(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 90 * 10010
    args.setting_name = 'semi_mt.mt_p05_res18_s2'
    return args


def mt_p06_s0(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 100 * 10010
    args.setting_name = 'semi_mt.mt_p06_res18'
    return args


def mt_p06_s1(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 100 * 10010
    args.setting_name = 'semi_mt.mt_p06_res18_s1'
    return args


def mt_p06_s2(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 100 * 10010
    args.setting_name = 'semi_mt.mt_p06_res18_s2'
    return args


def mt_p10_s0(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 100 * 10010
    args.setting_name = 'semi_mt.mt_part10_res18'
    return args


def mt_p10_s1(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 100 * 10010
    args.setting_name = 'semi_mt.mt_p10_res18_s1'
    return args


def mt_p10_s2(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 100 * 10010
    args.setting_name = 'semi_mt.mt_p10_res18_s2'
    return args


def mt_p20_s0(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 100 * 10010
    args.setting_name = 'semi_mt.mt_p20_res18'
    return args


def mt_p20_s1(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 100 * 10010
    args.setting_name = 'semi_mt.mt_p20_res18_s1'
    return args


def mt_p20_s2(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 100 * 10010
    args.setting_name = 'semi_mt.mt_p20_res18_s2'
    return args


def mt_p50_s0(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 90 * 10010
    args.setting_name = 'semi_mt.mt_p50_res18'
    return args


def mt_p50_s1(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 100 * 10010
    args.setting_name = 'semi_mt.mt_p50_res18_s1'
    return args


def mt_p50_s2(args):
    args.layers = 'encode_9'
    args.model_type = 'mt_vm_model'
    args.load_step = 100 * 10010
    args.setting_name = 'semi_mt.mt_p50_res18_s2'
    return args
