import nf_exp_settings.pat_basic_setting as pat_set
import nf_exp_settings.shared_settings as shr_set


def i2n_setting(args):
    args.objectome_zip = True
    args.gen_features = 1
    args.batchsize = 60
    args.image_prefix = '/mnt/fs4/chengxuz/v4it_temp_results/objectome'
    return args


def cate_seed0_res18(args):
    args = pat_set.cate_seed0_res18_basic(args)
    args = i2n_setting(args)
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/cate_res18_obj/seed0.hdf5'
    return args


def cate_seed1_res18(args):
    args = pat_set.cate_seed1_res18_basic(args)
    args = i2n_setting(args)
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/cate_res18_obj/seed1.hdf5'
    return args


def cate_seed2_res18(args):
    args = pat_set.cate_seed2_res18_basic(args)
    args = i2n_setting(args)
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/cate_res18_obj/seed2.hdf5'
    return args


def cate_seed3_res18(args):
    args = pat_set.cate_seed3_res18_basic(args)
    args = i2n_setting(args)
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/cate_res18_obj/seed3.hdf5'
    return args


def llp_dali_rj_setting(args):
    args.loadport = 28000
    args.loaddbname = 'aggre_semi'
    args.loadcolname = 'dyn_clstr_aux_dali'

    args.inst_model = 'all_spatial'
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'
    return args


def llp_p01_rj_setting(args):
    args = shr_set.basic_setting(args)
    args = shr_set.basic_res18_setting(args)
    args = i2n_setting(args)
    args = llp_dali_rj_setting(args)

    args.loadexpId = 'p01_tp10_ft'
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/llp_res18_obj/llp_p01_rj.hdf5'
    return args


def llp_p03_dali_rj_setting(args):
    args = shr_set.basic_setting(args)
    args = shr_set.basic_res18_setting(args)
    args = i2n_setting(args)
    args = llp_dali_rj_setting(args)

    args.loadexpId = 'p03_tp10_dali_ft'
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/llp_res18_obj/llp_p03_dali_rj.hdf5'
    return args


def llp_p05_dali_rj_setting(args):
    args = shr_set.basic_setting(args)
    args = shr_set.basic_res18_setting(args)
    args = i2n_setting(args)
    args = llp_dali_rj_setting(args)

    args.loadexpId = 'p05_tp10_dali_ft'
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/llp_res18_obj/llp_p05_dali_rj.hdf5'
    return args


def llp_p10_dali_rj_setting(args):
    args = shr_set.basic_setting(args)
    args = shr_set.basic_res18_setting(args)
    args = i2n_setting(args)
    args = llp_dali_rj_setting(args)

    args.loadport = 29000
    args.loadexpId = 'p10_tp10_dali_ctl'
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/llp_res18_obj/llp_p10_dali_rj.hdf5'
    return args
