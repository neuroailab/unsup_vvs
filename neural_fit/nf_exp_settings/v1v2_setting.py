from nf_exp_settings.shared_settings import \
        basic_color_bn_setting, basic_res18_setting, basic_mt_res18_setting


def v1v2_basic_setting_wo_db(args):
    args.whichopt = 1
    args.weight_decay = 1e-2
    args.batchsize = 30
    args.use_dataset_inter = True
    args.fre_valid = 100
    args.fre_metric = 100
    args.dataset_type = 'v1v2'
    args.val_on_train = 1
    return args


def v1v2_basic_setting(args):
    args = v1v2_basic_setting_wo_db(args)
    args.loadport = 27009
    args.nport = 27009
    args.dbname = 'new-neuralfit-v1v2'
    return args


def cate_load_setting(args):
    args.load_train_setting_func = 'part3_cate_res18'
    step_num = 64 * 10009
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/res18/model.ckpt-%i' % step_num
    args.train_num_steps = step_num + 15000
    return args


def cate_res18_v1v2(args):
    args = v1v2_basic_setting(args)
    args = basic_res18_setting(args)
    args = cate_load_setting(args)

    args.expId = 'cate_res18_v1v2_1'
    return args


def cate_res18_v1v2_swd(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_swd'
    args.weight_decay = 1e-3
    return args


def cate_res18_v1v2_bwd(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_bwd'
    args.weight_decay = 1e-1
    return args


def cate_res18_v1v2_bbwd(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_bbwd'
    args.weight_decay = 1.
    return args


def cate_res18_v1v2_bbbwd(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_bbbwd'
    args.weight_decay = 10.
    return args


def cate_res18_v1v2_l1(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_l1'
    args.weight_decay_type = 'l1'
    return args


def cate_res18_v1v2_l1_bwd(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_l1_bwd'
    args.weight_decay = 1e-1
    args.weight_decay_type = 'l1'
    return args


def cate_res18_v1v2_l1_bbwd(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_l1_bbwd'
    args.weight_decay = 1.
    args.weight_decay_type = 'l1'
    return args


def cate_res18_v1v2_l1_swd(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_l1_swd'
    args.weight_decay = 1e-3
    args.weight_decay_type = 'l1'
    return args


def cate_res18_v1v2_e1_bfpl_exp(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_e1_bfpl_exp'
    args.weight_decay = 1e-1
    args.spatial_select = '7,14'
    args.it_nodes = 'encode_1_bfpl_[0:1:14]_[0:1:14]'
    args.v4_nodes = 'encode_1_bfpl_[0:1:14]_[0:1:14]'
    return args


def cate_res18_v1v2_e1_exp(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_e1_exp'
    args.weight_decay = 1e-1
    args.spatial_select = '7,14'
    args.it_nodes = 'encode_1_[0:1:7][0:1:7]'
    args.v4_nodes = 'encode_1_[0:1:7][0:1:7]'
    return args


def cate_res18_v1v2_e2_exp(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_e2_exp'
    args.weight_decay = 1e-1
    args.spatial_select = '7,14'
    args.it_nodes = 'encode_2_[0:1:7][0:1:7]'
    args.v4_nodes = 'encode_2_[0:1:7][0:1:7]'
    return args


def cate_res18_v1v2_e3_exp(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_e3_exp'
    args.weight_decay = 1e-1
    args.spatial_select = '7,14'
    args.it_nodes = 'encode_3_[0:1:7][0:1:7]'
    args.v4_nodes = 'encode_3_[0:1:7][0:1:7]'
    return args


def cate_res18_v1v2_e4_exp(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_e4_exp'
    args.weight_decay = 1e-1
    args.spatial_select = '7,14'
    args.it_nodes = 'encode_4_[0:1:3][0:1:3]'
    args.v4_nodes = 'encode_4_[0:1:3][0:1:3]'
    return args


def cate_res18_v1v2_e5_exp(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_e5_exp'
    args.weight_decay = 1e-1
    args.spatial_select = '7,14'
    args.it_nodes = 'encode_5_[0:1:3][0:1:3]'
    args.v4_nodes = 'encode_5_[0:1:3][0:1:3]'
    return args


def cate_res18_v1v2_e4_exp_mre(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_e4_exp_mre'
    args.weight_decay = 1e-1
    args.spatial_select = '2,14'
    args.it_nodes = 'encode_4_[0:1:7]_[0:1:7]'
    args.v4_nodes = 'encode_4_[0:1:7]_[0:1:7]'
    return args


def cate_res18_v1v2_e5_exp_mre(args):
    args = cate_res18_v1v2(args)

    args.expId = 'cate_res18_v1v2_e5_exp_mre'
    args.weight_decay = 1e-1
    args.spatial_select = '2,14'
    args.it_nodes = 'encode_5_[0:1:7]_[0:1:7]'
    args.v4_nodes = 'encode_5_[0:1:7]_[0:1:7]'
    return args


def cate_res18_v1v2_bwd_to_be_filled(args):
    args = v1v2_basic_setting(args)
    args = basic_res18_setting(args)
    args = cate_load_setting(args)

    args.it_nodes = 'encode_[6:1:10]'
    args.v4_nodes = 'encode_[6:1:10]'
    args.weight_decay = 1e-1
    return args


def cate_res18_v1v2_bbwd_to_be_filled(args):
    args = cate_res18_v1v2_bwd_to_be_filled(args)
    args.weight_decay = 1.
    return args


def cate_res18_v1v2_e1_to_be_filled(args):
    args = cate_res18_v1v2_bwd_to_be_filled(args)
    args.spatial_select = '7,14'
    args.it_nodes = 'encode_1_[0:1:7]_[0:1:7]'
    args.v4_nodes = 'encode_1_[0:1:7]_[0:1:7]'
    return args


def cate_res18_v1v2_e2_to_be_filled(args):
    args = cate_res18_v1v2_bwd_to_be_filled(args)
    args.spatial_select = '7,14'
    args.it_nodes = 'encode_2_[0:1:7]_[0:1:7]'
    args.v4_nodes = 'encode_2_[0:1:7]_[0:1:7]'
    return args


def cate_res18_v1v2_e3_to_be_filled(args):
    args = cate_res18_v1v2_bwd_to_be_filled(args)
    args.spatial_select = '7,14'
    args.it_nodes = 'encode_3_[0:1:7]_[0:1:7]'
    args.v4_nodes = 'encode_3_[0:1:7]_[0:1:7]'
    return args


def cate_res18_v1v2_e4_to_be_filled(args):
    args = cate_res18_v1v2_bwd_to_be_filled(args)
    args.spatial_select = '7,14'
    args.it_nodes = 'encode_4_[0:1:3]_[0:1:3]'
    args.v4_nodes = 'encode_4_[0:1:3]_[0:1:3]'
    return args


def cate_res18_v1v2_e5_to_be_filled(args):
    args = cate_res18_v1v2_bwd_to_be_filled(args)
    args.spatial_select = '7,14'
    args.it_nodes = 'encode_5_[0:1:3]_[0:1:3]'
    args.v4_nodes = 'encode_5_[0:1:3]_[0:1:3]'
    return args
