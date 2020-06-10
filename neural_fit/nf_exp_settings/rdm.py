import nf_exp_settings.pat_basic_setting as pat_set
import nf_exp_settings.shared_settings as shr_set


def rdm_setting(args):
    args.h5py_data_loader = True
    args.gen_features = 1
    args.batchsize = 40
    return args


def cate_seed0_res18(args):
    args = pat_set.cate_seed0_res18_basic(args)
    args = rdm_setting(args)
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/cate_res18_rdm/seed0.hdf5'
    return args


def cate_seed1_res18(args):
    args = pat_set.cate_seed1_res18_basic(args)
    args = rdm_setting(args)
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/cate_res18_rdm/seed1.hdf5'
    return args


def cate_seed2_res18(args):
    args = pat_set.cate_seed2_res18_basic(args)
    args = rdm_setting(args)
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/cate_res18_rdm/seed2.hdf5'
    return args


def cate_seed3_res18(args):
    args = pat_set.cate_seed3_res18_basic(args)
    args = rdm_setting(args)
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/cate_res18_rdm/seed3.hdf5'
    return args


def la_orig_res18(args):
    args = pat_set.la_res18_basic(args)
    args = rdm_setting(args)
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/la_res18_rdm/orig.hdf5'
    return args


def la_new_res18(args):
    args = pat_set.la_res18_basic(args)
    args = rdm_setting(args)
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/la_res18_rdm/new.hdf5'
    args.ckpt_file = '/mnt/fs6/honglinc/'\
            + 'tf_experiments/models/la_multi_modal/'\
            + 'test/res18_LA_RGB/model.ckpt-2001800'
    return args
