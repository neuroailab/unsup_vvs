from nf_exp_settings.shared_settings import basic_setting, \
        basic_color_bn_setting, basic_res18_setting, basic_mt_res18_setting


def part3_cate_res18(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)

    args.expId = 'nf_part3_cate_res18_2'
    args.load_train_setting_func = 'part3_cate_res18'
    args.loadstep = 390390
    args.train_num_steps = 420390
    return args


def mt_part3_res18(args):
    args = basic_setting(args)
    args = basic_mt_res18_setting(args)

    #args.expId = 'nf_mt_part3_res18'
    args.expId = 'nf_mt_part3_res18_fx'
    args.load_train_setting_func = 'mt_part3_res18'
    args.loadstep = 1161160
    args.train_num_steps = 1191160
    return args


def mt_part3_res18_bbwd(args):
    args = mt_part3_res18(args)

    args.weight_decay = 1e-1
    args.expId = 'mt_part3_res18_bbwd'
    return args


def mt_part3_res18_bbbwd(args):
    args = mt_part3_res18(args)

    args.weight_decay = 1.
    args.expId = 'mt_part3_res18_bbbwd'
    return args


def mt_part3_res18_bwd(args):
    args = mt_part3_res18(args)

    args.weight_decay = 1e-2
    args.expId = 'mt_part3_res18_bwd'
    return args


def mt_part3_res18_swd(args):
    args = mt_part3_res18(args)

    args.weight_decay = 1e-4
    args.expId = 'mt_part3_res18_swd'
    return args


def mt_part3_res18_sswd(args):
    args = mt_part3_res18(args)

    args.weight_decay = 1e-5
    args.expId = 'mt_part3_res18_sswd'
    return args


def mt_part3_res18_inst(args):
    args = basic_setting(args)
    args = basic_mt_res18_setting(args)

    args.expId = 'nf_mt_part3_res18_inst'
    args.load_train_setting_func = 'mt_part3_res18_inst'
    args.loadstep = 730730
    args.train_num_steps = 760730
    return args


def inst_mt_res18_fst_p3(args):
    args = basic_setting(args)
    args = basic_mt_res18_setting(args)

    args.expId = 'nf_inst_mt_res18_fst_p3'
    args.load_train_setting_func = 'inst_mt_res18_fst_p3'
    args.loadstep = 640640
    args.train_num_steps = 670640
    return args


def inst_mt_res18_fst_p3_from_inst_fx(args):
    args = basic_setting(args)
    args = basic_mt_res18_setting(args)

    args.expId = 'nf_it_mt_res18_p3_fi'
    args.load_train_setting_func = 'inst_mt_res18_fst_p3_from_inst_fx'
    args.loadstep = 460460
    args.train_num_steps = 490460
    return args


def part5_cate_res18(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)

    args.expId = 'nf_part5_cate_res18'
    args.load_train_setting_func = 'part5_cate_res18'
    args.loadstep = 480480
    args.train_num_steps = 510480
    return args


def inst_cate_diff_check(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)

    args.expId = 'nf_inst_cate_diff_check'
    args.train_num_steps = 1226200
    args.load_train_setting_func = 'inst_and_cate_res18'
    args.loadstep = 1201200
    return args


def inst_cate_early(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)

    args.expId = 'nf_inst_cate_early'
    args.train_num_steps = 995970
    args.load_train_setting_func = 'inst_and_cate_res18_early'
    return args


def inst_cate_even_early(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)

    args.expId = 'nf_inst_cate_even_early'
    args.train_num_steps = 995970
    args.load_train_setting_func = 'inst_and_cate_res18_even_early'
    return args


def basic_inst_model_setting(args):
    args.inst_model = 'all_spatial'
    args.loaddbname = 'instance_task'
    args.loadcolname = 'dynamic_clustering'
    return args


def inst_model_km_or_mre(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)

    args.expId = 'inst_model_km_or_mre'
    args.loadexpId = 'dynamic_km_or_mre'
    args.loadstep = 1961764
    args.train_num_steps = 1991764
    return args


def semi_p03_nf(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)

    args.expId = 'semi_p03_nf'
    args.loadexpId = 'dynamic_semi_p03_3'
    args.loadstep = 3423078
    args.train_num_steps = 3453078
    return args


def obj_setting(args):
    args.objectome_zip = True
    args.gen_features = 1
    args.batchsize = 60
    args.image_prefix = '/home/chengxuz/objectome'
    return args


def inst_model_km_or_mre_obj(args):
    args = inst_model_km_or_mre(args)
    args = obj_setting(args)

    args.gen_hdf5path = '/home/chengxuz/inst_model_km_or_mre_obj.hdf5'
    args.it_nodes = 'encode_[2:1:10]'
    args.v4_nodes = 'encode_[2:1:10]'
    return args


def inst_model_km_or_mre_bbbwd(args):
    args = inst_model_km_or_mre(args)
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'

    args.expId = 'inst_model_km_or_mre_bbbwd'
    args.weight_decay = 1.
    return args


def inst_model_km_or_mre_bbwd(args):
    args = inst_model_km_or_mre(args)
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'

    args.expId = 'inst_model_km_or_mre_bbwd'
    args.weight_decay = 1e-1
    return args


def inst_model_km_or_mre_bwd(args):
    args = inst_model_km_or_mre(args)
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'

    args.expId = 'inst_model_km_or_mre_bwd'
    args.weight_decay = 1e-2
    return args


def inst_model_km_or_mre_swd(args):
    args = inst_model_km_or_mre(args)
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'

    args.expId = 'inst_model_km_or_mre_swd'
    args.weight_decay = 1e-4
    return args


def inst_model_km_or_mre_sswd(args):
    args = inst_model_km_or_mre(args)
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'

    args.expId = 'inst_model_km_or_mre_sswd'
    args.weight_decay = 1e-5
    return args


def mid_setting(args):
    args.loaddbname = 'instance_transfer'
    args.loadcolname = 'dyn_clustering'

    args.inst_model = 'Mid-500,1000,1500,2000'
    args.it_nodes = 'mid500,mid1000,mid1500,mid2000'
    args.v4_nodes = 'mid500,mid1000,mid1500,mid2000'
    return args


def inst_model_mre_mid_obj(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = obj_setting(args)
    args = mid_setting(args)

    args.loadexpId = 'dyn_km_ormre_mid'
    args.gen_hdf5path = '/home/chengxuz/inst_model_mre_mid_obj.hdf5'
    return args


def semi_p03_mid_obj(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = obj_setting(args)
    args = mid_setting(args)

    args.loadexpId = 'semi_p03_mid'
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/semi_p03_mid_obj.hdf5'
    return args


def llp_setting(args):
    args.loadport = 27006
    args.loaddbname = 'aggre_semi'
    args.loadcolname = 'dyn_clstr'

    args.inst_model = 'all_spatial'
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'
    return args


def nf_semi_p03_tp10_wc_cf_lclw(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = llp_setting(args)

    args.loadexpId = 'p03_tp10_wc_cf_lclw'
    args.expId = 'nf_llp_p03'
    args.loadstep = 4153735
    args.train_num_steps = 4183735
    return args


def nf_semi_p03_tp10_wc_cf_lclw_bbbwd(args):
    args = nf_semi_p03_tp10_wc_cf_lclw(args)

    args.expId = 'nf_llp_p03_bbbwd'
    args.weight_decay = 1.
    return args


def nf_semi_p03_tp10_wc_cf_lclw_bbwd(args):
    args = nf_semi_p03_tp10_wc_cf_lclw(args)

    args.expId = 'nf_llp_p03_bbwd'
    args.weight_decay = 1e-1
    return args


def nf_semi_p03_tp10_wc_cf_lclw_bwd(args):
    args = nf_semi_p03_tp10_wc_cf_lclw(args)

    args.expId = 'nf_llp_p03_bwd'
    args.weight_decay = 1e-2
    return args


def nf_semi_p03_tp10_wc_cf_lclw_swd(args):
    args = nf_semi_p03_tp10_wc_cf_lclw(args)

    args.expId = 'nf_llp_p03_swd'
    args.weight_decay = 1e-4
    return args


def nf_semi_p03_tp10_wc_cf_lclw_sswd(args):
    args = nf_semi_p03_tp10_wc_cf_lclw(args)

    args.expId = 'nf_llp_p03_sswd'
    args.weight_decay = 1e-5
    return args


def semi_p03_tp10_wc_cf_lclw_obj(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = obj_setting(args)
    args = llp_setting(args)

    args.loadexpId = 'p03_tp10_wc_cf_lclw'
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/semi_p03_tp10_wc_cf_lclw_obj.hdf5'
    return args


def semi_p05_tp10_wc_cf_lclw_obj(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = obj_setting(args)
    args = llp_setting(args)

    args.loadexpId = 'p05_tp10_wc_cf_lclw'
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/semi_p05_tp10_wc_cf_lclw_obj.hdf5'
    return args


def semi_p10_tp10_wc_cf_lclw_obj(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = obj_setting(args)
    args = llp_setting(args)

    args.loadexpId = 'p10_tp10_wc_cf_lclw'
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/semi_p10_tp10_wc_cf_lclw_obj.hdf5'
    return args


def mt_part3_res18_obj(args):
    args = basic_setting(args)
    args = basic_mt_res18_setting(args)
    args = obj_setting(args)

    args.load_train_setting_func = 'mt_part3_res18'
    args.loadstep = 1161160
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/semi_mt_p03_obj.hdf5'
    return args


def mt_part5_res18_obj(args):
    args = basic_setting(args)
    args = basic_mt_res18_setting(args)
    args = obj_setting(args)

    args.load_train_setting_func = 'mt_part5_res18'
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/semi_mt_p05_obj.hdf5'
    return args


def mt_part10_res18_obj(args):
    args = basic_setting(args)
    args = basic_mt_res18_setting(args)
    args = obj_setting(args)

    args.load_train_setting_func = 'mt_part10_res18_fx'
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/semi_mt_p10_obj.hdf5'
    return args


def part3_cate_res18_obj(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = obj_setting(args)

    args.load_train_setting_func = 'part3_cate_res18'
    args.loadstep = 390390
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v4it_temp_results/ctl_p03_obj.hdf5'
    return args


def basic_inst_ctl_model_setting(args):
    args.inst_model = 'all_spatial'
    args.loaddbname = 'instance_task'
    args.loadcolname = 'control'
    args.loadexpId = 'full'
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'
    return args


def inst_ctl_model(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = basic_inst_ctl_model_setting(args)

    args.expId = 'inst_ctl_model'
    args.loadstep = 2342106
    args.train_num_steps = 2372106
    return args


def inst_ctl_model_swd(args):
    args = inst_ctl_model(args)

    args.weight_decay = 1e-4
    args.expId = 'inst_ctl_model_swd'
    return args


def inst_ctl_model_sswd(args):
    args = inst_ctl_model(args)

    args.weight_decay = 1e-5
    args.expId = 'inst_ctl_model_sswd'
    return args


def inst_ctl_model_bwd(args):
    args = inst_ctl_model(args)

    args.weight_decay = 1e-2
    args.expId = 'inst_ctl_model_bwd'
    return args


def inst_ctl_model_bbwd(args):
    args = inst_ctl_model(args)

    args.weight_decay = 1e-1
    args.expId = 'inst_ctl_model_bbwd'
    return args


def inst_ctl_model_bbbwd(args):
    args = inst_ctl_model(args)

    args.weight_decay = 1.
    args.expId = 'inst_ctl_model_bbbwd'
    return args


def cate_res18_to_be_filled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)

    args.load_train_setting_func = 'part3_cate_res18'
    args.weight_decay = 1e-2
    return args


def cate_res18_obj_to_be_filled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = obj_setting(args)

    args.load_train_setting_func = 'part3_cate_res18'
    args.v4_nodes = 'encode_9'
    args.it_nodes = 'encode_9'
    return args


def llp_p03_to_be_filled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = llp_setting(args)

    args.loadexpId = 'p03_tp10_wc_cf_lclw'
    args.weight_decay = 1e-2
    return args


def llp_p03_obj_to_be_filled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = llp_setting(args)

    args.loadexpId = 'p03_tp10_wc_cf_lclw'
    args = obj_setting(args)
    args.v4_nodes = 'encode_10'
    args.it_nodes = 'encode_10'
    return args


def inst_ctl_model_to_be_filled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = basic_inst_ctl_model_setting(args)
    args.weight_decay = 1e-2
    return args


def inst_ctl_model_obj_to_be_filled(args):
    args = basic_setting(args)
    args = basic_res18_setting(args)
    args = basic_inst_ctl_model_setting(args)

    args = obj_setting(args)
    args.v4_nodes = 'encode_10'
    args.it_nodes = 'encode_10'
    return args


def deepcluster_res18(args):
    args = basic_setting(args)
    args = basic_color_bn_setting(args)

    args.weight_decay = 1e-2
    args.deepcluster = 'res18'
    args.train_num_steps = 20000
    args.it_nodes = 'conv[105:1:109]'
    args.v4_nodes = 'conv[105:1:109]'
    args.expId = 'dc_res18'
    return args
