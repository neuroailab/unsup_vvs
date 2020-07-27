import shared_settings


def basic_semi_setting(args):
    args.init_type = 'instance_resnet'
    args.tpu_flag = 1
    args.weight_decay = 1e-4
    args.whichopt = 0
    args.init_lr = 0.03
    args.imgnt_w_idx = True
    args.dbname = 'semi_instance'
    args.network_func = 'get_resnet_18'
    args.network_func_kwargs = '{"num_cat": 128}'
    return args


def p03_clstr_setting(args):
    args.whichimagenet = 'part3_widx_ordrd'
    args.dataconfig = 'dataset_task_config_image_clst.json'
    args.inst_clstr_path = '/mnt/fs3/divyansh/imagenet_alex_p03.npy'
    return args


def p03_clstr_un_semi_setting(args):
    args.whichimagenet = 'part3_widx_ordrd'
    args.dataconfig = 'dataset_task_config_image_clst_un_semi.json'
    args.inst_clstr_path = '/mnt/fs3/divyansh/imagenet_alex_p03.npy'
    args.semi_clstr_path = '/mnt/fs3/divyansh/imagenet_alex_p03.npy'
    return args


def p03_clstr_un_mean_teacher_setting(args):
    args.whichimagenet = 'part3_widx_ordrd'
    args.dataconfig = 'dataset_task_config_image_clst_un_mt.json'
    args.inst_clstr_path = '/mnt/fs3/divyansh/imagenet_alex_p03.npy'
    return args


def imagenet_p03_clstr(args):
    args = shared_settings.basic_setting(args)
    args = shared_settings.bs_128_setting(args)
    args.fre_cache_filter = 10010
    args.fre_filter = 100100
    args = basic_semi_setting(args)
    args = p03_clstr_setting(args)
    
    args.expId = 'imagenet_p03_clstr_2'
    args.lr_boundaries = "1753017,2054102"
    return args


def imagenet_p03_clstr_0k(args):
    args = shared_settings.basic_setting(args)
    args = shared_settings.bs_128_setting(args)
    args.fre_cache_filter = 10010
    args.fre_filter = 100100
    args = basic_semi_setting(args)
    args = p03_clstr_setting(args)
    
    args.expId = 'imagenet_p03_clstr_0k'
    args.instance_k = 0
    args.lr_boundaries = "595764"
    return args


def p03_clstr_un_inst(args):
    args = shared_settings.basic_setting(args)
    args = shared_settings.bs_128_setting(args)
    args.fre_cache_filter = 10010
    args.fre_filter = 100100
    args = basic_semi_setting(args)
    args = p03_clstr_setting(args)
    args.dataconfig = "dataset_task_config_image_clst_un_inst.json"
    args.network_func = 'get_resnet_18_wun'
    
    args.expId = 'p03_clstr_un_inst'
    return args


def p03_clstr_un_inst_only_mdl(args):
    args = shared_settings.basic_setting(args)
    args = shared_settings.bs_128_setting(args)
    args.fre_cache_filter = 10010
    args.fre_filter = 50050
    args = basic_semi_setting(args)
    args = p03_clstr_setting(args)
    args.dataconfig = "dataset_task_config_image_clst_un_inst_only_model.json"
    args.network_func = 'get_resnet_18_wun'
    
    args.expId = 'p03_clstr_un_inst_only_mdl'
    return args


def p03_clstr_un_semi(args):
    args = shared_settings.basic_setting(args)
    args = shared_settings.bs_128_setting(args)
    args.fre_cache_filter = 10010
    args.fre_filter = 100100
    args = basic_semi_setting(args)
    args = p03_clstr_un_semi_setting(args)
    args.network_func = 'get_resnet_18_wun'
    
    args.expId = 'p03_clstr_un_semi3'
    args.instance_k = 0
    #args.loadexpId = 'p03_clstr_un_inst'
    #args.loadstep = 100100
    #args.semi_name_scope = 'imagenet'
    return args


def p03_clstr_semi_frm_dset(args):
    args = shared_settings.basic_setting(args)
    args = shared_settings.bs_128_setting(args)
    args.fre_cache_filter = 10010
    args.fre_filter = 100100
    args = basic_semi_setting(args)
    args = p03_clstr_un_semi_setting(args)
    args.network_func = 'get_resnet_18_wun'
    
    args.expId = 'p03_clstr_semi_frm_dset'
    args.loadexpId = 'inst_res18_newp_dset'
    args.loadstep = 100000
    args.load_dbname = 'combinet-test'
    args.loadport = 27006
    args.semi_name_scope = 'imagenet'
    return args


def p03_clstr_un_mt(args):
    args = shared_settings.basic_setting(args)
    args = shared_settings.bs_128_setting(args)
    args.fre_cache_filter = 10010
    args.fre_filter = 50050
    args = basic_semi_setting(args)
    args = p03_clstr_un_mean_teacher_setting(args)
    args.network_func = 'get_res18_clstr_un_mt'
    
    args.expId = 'p03_clstr_un_mt'
    return args


def p03_clstr_un_mt_model_setting(args):
    args.whichimagenet = 'part3_widx_ordrd'
    args.dataconfig = 'dataset_task_config_image_clst_un_mt_mdl.json'
    args.inst_clstr_path = '/mnt/fs3/divyansh/imagenet_alex_p03.npy'
    args.mt_ramp_down = 1
    args.res_coef = 1
    return args


def p03_clstr_un_mt_mdl(args):
    args = shared_settings.basic_setting(args)
    args = shared_settings.bs_128_setting(args)
    args.fre_cache_filter = 10010
    args.fre_filter = 50050
    args = basic_semi_setting(args)
    args = p03_clstr_un_mt_model_setting(args)
    args.network_func = 'get_res18_clstr_un_mt'
    
    args.expId = 'p03_clstr_un_mt_mdl'
    return args
