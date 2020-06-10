from nf_exp_settings.shared_settings import basic_res18_setting, basic_inst_model_setting, \
        convrnn_pretrain_setting, basic_convrnn_model_setting, basic_setting, \
        basic_color_bn_setting


def convrnn_pretrain_hvm(args):
    args = basic_setting(args)
    args = basic_color_bn_setting(args)
    args = basic_convrnn_model_setting(args)
    args = convrnn_pretrain_setting(args)

    args.expId = 'convrnn_pretrain_hvm'
    args.weight_decay = 1e-2
    args.img_out_size = 256
    args.train_num_steps = 240105
    return args


def convrnn_pretrain_hvm_swd(args):
    args = convrnn_pretrain_hvm(args)
    args.expId = 'convrnn_pretrain_hvm_swd'
    args.weight_decay = 1e-3
    return args


def convrnn_pretrain_hvm_bwd(args):
    args = convrnn_pretrain_hvm(args)
    args.expId = 'convrnn_pretrain_hvm_bwd'
    args.weight_decay = 1e-1
    return args


def depth_res18_load(args):
    args.pathconfig = 'depth_resnet18_up4.cfg'
    step_num = 2982682
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/depth_res18/model.ckpt-%i' % step_num
    args.train_num_steps = step_num + 30000
    args.ignorebname_new = 0
    args.batch_name = '_pbrnet'
    args.div_std = 0
    args.rp_sub_mean = 0
    return args


def depth_res18_hvm(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = depth_res18_load(args)

    args.weight_decay = 1e-2
    #args.expId = 'depth_res18_hvm'
    args.expId = 'depth_res18_hvm_fx'
    return args


def rp_res18_load(args):
    args.pathconfig = 'rp_resnet18.cfg'
    step_num = 1181162
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/rp_res18/model.ckpt-%i' % step_num
    args.train_num_steps = step_num + 30000
    args.ignorebname_new = 0
    args.batch_name = '_rp'
    args.div_std = 0
    return args


def rp_res18_hvm(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = rp_res18_load(args)
    args.weight_decay = 1e-2
    args.expId = 'rp_res18_hvm_fxstd'
    return args


def col_res18_load(args):
    args.pathconfig = 'depth_resnet18_up4.cfg'
    step_num = 5605040
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18/model.ckpt-%i' % step_num
    args.train_num_steps = step_num + 30000
    args.ignorebname_new = 0
    args.batch_name = '_colorization'
    args.div_std = 0
    args.rp_sub_mean = 0
    args.color_prep = 1
    return args


def col_res18_hvm(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = col_res18_load(args)

    #args.expId = 'col_res18_hvm'
    args.expId = 'col_res18_hvm_fx'
    args.weight_decay = 1e-2
    return args


def col_dilat_res18_hvm(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = col_res18_load(args)

    args.pathconfig = 'col_resnet18.cfg'
    args.expId = 'col_dilat_res18_hvm' # wrong result there
    return args
