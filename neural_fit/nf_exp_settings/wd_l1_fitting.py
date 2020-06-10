from nf_exp_settings.shared_settings import basic_res18_setting, \
        basic_inst_model_setting
import nf_exp_settings.v1_cadena_setting as v1_sts


def wd_l1_save(args):
    args.nport = 27009
    args.dbname = 'wd-l1-fitting'
    args.colname = 'pat'
    return args


def cate_res18_l1_basic(args):
    args = wd_l1_save(args)
    args = basic_res18_setting(args)
    args = v1_sts.cate_res18_v1_load(args)

    args.weight_decay_type = 'l1'
    args.expId = 'cate_res18'
    return args


def la_res18_l1_basic(args):
    args = wd_l1_save(args)
    args = basic_inst_model_setting(args)
    args = v1_sts.la_res18_v1_load(args)

    args.loadport = 27009
    args.it_nodes = 'encode_[2:1:11]'
    args.weight_decay_type = 'l1'
    args.expId = 'la_res18'
    return args


def untrn_res18_l1_basic(args):
    args = wd_l1_save(args)
    args = basic_res18_setting(args)

    args.loadport = 27009
    args.loaddbname = 'wd-l1-fitting'
    args.loadcolname = 'pat'
    args.network_func = 'get_resnet_18'
    args.weight_decay_type = 'l1'
    args.expId = 'untrn_res18'
    args.train_num_steps = 30000
    return args
