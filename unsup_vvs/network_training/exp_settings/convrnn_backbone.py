import shared_settings


def convrnn_basic_setting(args):
    args = shared_settings.basic_setting(args)
    args = shared_settings.bs_128_setting(args)
    return args


def convrnn_cate(args):
    args = convrnn_basic_setting(args)
    args = shared_settings.basic_cate_setting(args) 
    args.network_func = 'convrnn_funcs.convrnn_cate'
    args.expId = 'convrnn_cate'
    args.lr_boundaries = "360300"
    return args
