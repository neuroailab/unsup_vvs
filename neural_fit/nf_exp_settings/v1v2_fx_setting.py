from nf_exp_settings.shared_settings import \
        basic_color_bn_setting, basic_res18_setting, basic_mt_res18_setting
from nf_exp_settings.v1v2_setting import v1v2_basic_setting, cate_load_setting
import nf_exp_settings.v1v2_setting as v1v2_setting


def cate_res18_v1v2_bwd_fx(args):
    args = v1v2_basic_setting(args)
    args = basic_res18_setting(args)
    args = cate_load_setting(args)

    args.expId = 'cate_res18_v1v2_bwd_fx'
    args.img_out_size = 128
    args.weight_decay = 1e-1
    return args


def cate_res18_v1v2_bbwd_fx(args):
    args = cate_res18_v1v2_bwd_fx(args)

    args.expId = 'cate_res18_v1v2_bbwd_fx'
    args.weight_decay = 1.
    return args


def cate_res18_v1v2_bwd_fx_to_be_filled(args):
    args = v1v2_basic_setting(args)
    args = basic_res18_setting(args)
    args = cate_load_setting(args)

    args.it_nodes = 'encode_[4:1:10]'
    args.v4_nodes = 'encode_[4:1:10]'
    args.img_out_size = 128
    args.weight_decay = 1e-1
    return args


def cate_res18_v1v2_bbwd_fx_to_be_filled(args):
    args = cate_res18_v1v2_bwd_fx_to_be_filled(args)
    args.weight_decay = 1.
    return args


def cate_res18_v1v2_fx_e1_to_be_filled(args):
    args = cate_res18_v1v2_bwd_fx_to_be_filled(args)
    args.spatial_select = '8,16'
    args.it_nodes = 'encode_1_[0:1:3]_[0:1:3]'
    args.v4_nodes = 'encode_1_[0:1:3]_[0:1:3]'
    return args


def cate_res18_v1v2_fx_e2_to_be_filled(args):
    args = cate_res18_v1v2_bwd_fx_to_be_filled(args)
    args.spatial_select = '8,16'
    args.it_nodes = 'encode_2_[0:1:3]_[0:1:3]'
    args.v4_nodes = 'encode_2_[0:1:3]_[0:1:3]'
    return args


def cate_res18_v1v2_fx_e3_to_be_filled(args):
    args = cate_res18_v1v2_bwd_fx_to_be_filled(args)
    args.spatial_select = '8,16'
    args.it_nodes = 'encode_3_[0:1:3]_[0:1:3]'
    args.v4_nodes = 'encode_3_[0:1:3]_[0:1:3]'
    return args


def cate_res18_v1v2_bwd_fx_crop(args):
    args = v1v2_basic_setting(args)
    args = basic_res18_setting(args)
    args = cate_load_setting(args)
    args.img_out_size = 128
    args.img_crop_size = 72

    args.expId = 'cate_res18_v1v2_bwd_fx_crop'
    args.weight_decay = 1e-1
    return args


def cate_res18_v1v2_bbwd_fx_crop(args):
    args = cate_res18_v1v2_bwd_fx_crop(args)

    args.expId = 'cate_res18_v1v2_bbwd_fx_crop'
    args.weight_decay = 1.
    return args


def v1v2_trandom_50to200_basic_setting_wo_db(args):
    args = v1v2_setting.v1v2_basic_setting_wo_db(args)
    args.img_out_size = 128
    args.img_crop_size = 72
    args.v1v2_folder = '/mnt/fs4/chengxuz/v1v2_related/tfrs_txt_random_50to200/'
    return args


def v1v2_pat_basic_save_setting(args):
    args.dbname = 'v1v2-rescue'
    args.nport = 27009
    args.colname = 'pat'
    return args


def cate_res18_basic(args):
    args = v1v2_pat_basic_save_setting(args)
    args = basic_res18_setting(args)
    args = cate_load_setting(args)
    args.expId = 'cate_res18'
    return args
