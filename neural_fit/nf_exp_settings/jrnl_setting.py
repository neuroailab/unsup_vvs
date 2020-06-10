from nf_exp_settings.shared_settings import basic_res18_setting, basic_inst_model_setting, \
        basic_setting, deepcluster_res18_deseq
from nf_exp_settings.v1_cadena_setting import v1_cadena_fit_setting, cate_res18_v1_load, \
        la_res18_v1_load, ir_res18_v1_load
from nf_exp_settings.v4it_hvm_setting import col_res18_load, rp_res18_load, depth_res18_load


def cate_res18_v1_tc_unfilled(args):
    args = basic_res18_setting(args)
    args = v1_cadena_fit_setting(args)
    args = cate_res18_v1_load(args)
    return args


def cate_res18_v1_tc_swd_unfilled(args):
    args = cate_res18_v1_tc_unfilled(args)
    args.weight_decay = 1e-3
    return args


def cate_res18_v4it_unfilled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = cate_res18_v1_load(args)
    args.weight_decay = 1e-2
    return args


def cate_res18_v4it_swd_unfilled(args):
    args = cate_res18_v4it_unfilled(args)
    args.weight_decay = 1e-3
    return args


def la_res18_v1_tc_unfilled(args):
    args = basic_res18_setting(args)
    args = v1_cadena_fit_setting(args)
    args = basic_inst_model_setting(args)
    args = la_res18_v1_load(args)
    return args


def la_res18_v1_tc_swd_unfilled(args):
    args = la_res18_v1_tc_unfilled(args)
    args.weight_decay = 1e-3
    return args


def la_res18_v4it_unfilled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = la_res18_v1_load(args)
    args.it_nodes = 'encode_[2:1:11]'
    args.weight_decay = 1e-2
    return args


def la_res18_v4it_swd_unfilled(args):
    args = la_res18_v4it_unfilled(args)
    args.weight_decay = 1e-3
    return args


def ir_res18_v1_tc_unfilled(args):
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = v1_cadena_fit_setting(args)
    args = ir_res18_v1_load(args)
    return args


def ir_res18_v1_tc_swd_unfilled(args):
    args = ir_res18_v1_tc_unfilled(args)
    args.weight_decay = 1e-3
    return args


def ir_res18_v4it_unfilled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = ir_res18_v1_load(args)
    args.weight_decay = 1e-2
    args.it_nodes = 'encode_[2:1:11]'
    return args


def ir_res18_v4it_swd_unfilled(args):
    args = ir_res18_v4it_unfilled(args)
    args.weight_decay = 1e-3
    return args


def llp_res18_v1_load(args):
    args.loadport = 27006
    args.loaddbname = 'aggre_semi'
    args.loadcolname = 'dyn_clstr'

    args.inst_model = 'all_spatial'
    args.v4_nodes = 'encode_[2:1:11]'
    args.loadexpId = 'p03_tp10_wc_cf_lclw'
    args.train_num_steps = 4183735
    return args


def llp_res18_v1_tc_unfilled(args):
    args = basic_res18_setting(args)
    args = v1_cadena_fit_setting(args)
    args = llp_res18_v1_load(args)
    return args


def llp_res18_v1_tc_swd_unfilled(args):
    args = llp_res18_v1_tc_unfilled(args)
    args.weight_decay = 1e-3
    return args


def llp_res18_v4it_unfilled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = llp_res18_v1_load(args)
    args.it_nodes = 'encode_[2:1:11]'
    args.weight_decay = 1e-2
    return args


def llp_res18_v4it_swd_unfilled(args):
    args = llp_res18_v4it_unfilled(args)
    args.weight_decay = 1e-3
    return args


def col_res18_v1_tc_unfilled(args):
    args = basic_res18_setting(args)
    args = v1_cadena_fit_setting(args)
    args = col_res18_load(args)
    args.weight_decay = 1e-2
    return args


def col_res18_v1_tc_swd_unfilled(args):
    args = col_res18_v1_tc_unfilled(args)
    args.weight_decay = 1e-3
    return args


def rp_res18_v1_tc_unfilled(args):
    args = basic_res18_setting(args)
    args = v1_cadena_fit_setting(args)
    args = rp_res18_load(args)
    args.weight_decay = 1e-2
    return args


def rp_res18_v1_tc_swd_unfilled(args):
    args = rp_res18_v1_tc_unfilled(args)
    args.weight_decay = 1e-3
    return args


def depth_res18_v1_tc_unfilled(args):
    args = basic_res18_setting(args)
    args = v1_cadena_fit_setting(args)
    args = depth_res18_load(args)
    args.weight_decay = 1e-2
    return args


def depth_res18_v1_tc_swd_unfilled(args):
    args = depth_res18_v1_tc_unfilled(args)
    args.weight_decay = 1e-3
    return args


def col_res18_v4it_unfilled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = col_res18_load(args)
    args.weight_decay = 1e-2
    return args


def col_res18_v4it_swd_unfilled(args):
    args = col_res18_v4it_unfilled(args)
    args.weight_decay = 1e-3
    return args


def rp_res18_v4it_unfilled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = rp_res18_load(args)
    args.weight_decay = 1e-2
    return args


def rp_res18_v4it_swd_unfilled(args):
    args = rp_res18_v4it_unfilled(args)
    args.weight_decay = 1e-3
    return args


def depth_res18_v4it_unfilled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = depth_res18_load(args)
    args.weight_decay = 1e-2
    return args


def depth_res18_v4it_swd_unfilled(args):
    args = depth_res18_v4it_unfilled(args)
    args.weight_decay = 1e-3
    return args


def dc_res18_v4it_unfilled(args):
    args = basic_setting(args)
    args = deepcluster_res18_deseq(args)
    args.weight_decay = 1e-2
    return args


def dc_res18_v4it_swd_unfilled(args):
    args = dc_res18_v4it_unfilled(args)
    args.weight_decay = 1e-3
    return args


def dc_res18_v1_tc_unfilled(args):
    args = v1_cadena_fit_setting(args)
    args = deepcluster_res18_deseq(args)
    args.it_nodes = None
    args.deepcluster = 'res18_deseq_v1'
    args.weight_decay = 1e-2
    return args


def dc_res18_v1_tc_swd_unfilled(args):
    args = dc_res18_v1_tc_unfilled(args)
    args.weight_decay = 1e-3
    return args


def untrn_res18_v1_tc_unfilled(args):
    args = basic_res18_setting(args)
    args = v1_cadena_fit_setting(args)
    args.loaddbname = 'new-neuralfit-v1'
    args.loadcolname = 'neuralfit'
    args.network_func = 'get_resnet_18'
    args.train_num_steps = 30000
    return args


def untrn_res18_v1_tc_swd_unfilled(args):
    args = untrn_res18_v1_tc_unfilled(args)
    args.weight_decay = 1e-3
    return args


def untrn_res18_v4it_unfilled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args.weight_decay = 1e-2
    args.loaddbname = 'new-neuralfit'
    args.loadcolname = 'neuralfit'
    args.network_func = 'get_resnet_18'
    args.train_num_steps = 30000
    return args


def untrn_res18_v4it_swd_unfilled(args):
    args = untrn_res18_v4it_unfilled(args)
    args.weight_decay = 1e-3
    return args


def ir_vm_res18_load(args):
    args.loadport = 27006
    args.loaddbname = 'combinet-test'
    args.loadcolname = 'combinet'

    args.loadexpId = 'inst_res18_newp_dset'
    args.loadstep = 2500000
    args.train_num_steps = 2530000
    args.pathconfig = 'instance_resnet18.cfg'
    return args


def ir_vm_res18_v1_tc_unfilled(args):
    args = basic_res18_setting(args)
    args = v1_cadena_fit_setting(args)
    args = ir_vm_res18_load(args)
    return args


def ir_vm_res18_v1_tc_swd_unfilled(args):
    args = ir_vm_res18_v1_tc_unfilled(args)
    args.weight_decay = 1e-3
    return args
