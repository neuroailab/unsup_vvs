from nf_exp_settings.shared_settings import bs_basic_setting, bs_res18_setting


def v1_cadena_setting(args):
    args.data_norm_type = 'v1_cadena'
    args.img_out_size = 40
    args.benchmark = 'ToliasCadena2017PLS'
    return args


def cate_res18_v1_cadena(args):
    args = bs_basic_setting(args)
    args = bs_res18_setting(args)
    args = v1_cadena_setting(args)

    step_num = 64 * 10009
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/res18/model.ckpt-%i' % step_num
    args.expId = 'cate_res18_v1_cadena'
    return args
