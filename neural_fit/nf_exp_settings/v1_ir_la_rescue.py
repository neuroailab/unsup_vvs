from nf_exp_settings.shared_settings import basic_res18_setting, \
        basic_inst_model_setting, \
        basic_setting, basic_color_bn_setting
from nf_exp_settings.v1_cadena_setting import \
        la_res18_v1_load, ir_res18_v1_load, v1_cadena_fit_setting


def v1_rescue_fit_setting(args):
    args = v1_cadena_fit_setting(args)
    args.dbname = 'v1-rescue'
    return args


def la_res18_v1_fit_100(args):
    args = basic_res18_setting(args)
    args = v1_rescue_fit_setting(args)
    args = basic_inst_model_setting(args)
    args = la_res18_v1_load(args, 1000900)
    args.expId = 'la_res18_v1_fit_100'
    return args


def ir_res18_v1_fit_100(args):
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = v1_rescue_fit_setting(args)
    args = ir_res18_v1_load(args, 1000900)
    args.expId = 'ir_res18_v1_fit_100'
    return args


def ir_cate_res18_v1_load(args, loadstep):
    args.loaddbname = 'combine_instance'
    args.loadcolname = 'combinet'
    args.v4_nodes = 'encode_[1:1:10]'
    args.loadexpId = 'res18_inst_cate_1'
    args.loadstep = loadstep
    args.train_num_steps = loadstep + 30000
    return args


def ir_cate_res18_v1_fit(args):
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = v1_rescue_fit_setting(args)
    args = ir_cate_res18_v1_load(args, 990990)
    args.expId = 'ir_cate_res18_v1_fit'
    return args


def cate_inst_prep_res18_v1_load(args, loadstep):
    args.loaddbname = 'cate_aug'
    args.loadcolname = 'res18'
    args.loadport = 27007
    args.v4_nodes = 'encode_[1:1:10]'
    args.loadexpId = 'inst_prep'
    args.loadstep = loadstep
    args.train_num_steps = loadstep + 30000
    return args


def cate_inst_prep_res18_v1_fit(args):
    args = basic_res18_setting(args)
    args = v1_rescue_fit_setting(args)
    args = cate_inst_prep_res18_v1_load(args, 490490)
    args.expId = 'cate_inst_prep_res18_v1_fit'
    args.load_train_setting_func = 'cate_res18_inst_prep'
    return args


def ml_load_setting(args):
    args.loadport = 27007
    args.loaddbname = 'multi_loss'
    args.loadcolname = 'res18'
    args.v4_nodes = 'encode_[2:1:11]'
    return args


def ml1_ir_res18_v1_load(args, loadstep):
    args = ml_load_setting(args)
    args.loadexpId = 'ml1_res18_IR'
    args.loadstep = loadstep
    args.train_num_steps = loadstep + 30000
    return args


def mll_ir_res18_v1_fit_100(args):
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = v1_rescue_fit_setting(args)
    args = ml1_ir_res18_v1_load(args, 1000900)
    args.expId = 'mll_ir_res18_v1_fit_100'
    return args


def ml1_ir_res18_v1_fit(args):
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = v1_rescue_fit_setting(args)
    args = ml1_ir_res18_v1_load(args, 10009 * 218)
    args.expId = 'ml1_ir_res18_v1_fit'
    return args


def ml1_ir_res18_v4it_fit(args):
    args = rescue_basic_setting(args)
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = ml1_ir_res18_v1_load(args, 10009 * 218)
    args.it_nodes = 'encode_[2:1:11]'
    args.expId = 'ml1_ir_res18_v4it_fit'
    args.weight_decay = 1e-2
    return args


def ml_ir_res18_v1_load(args, loadstep):
    args = ml_load_setting(args)
    args.loadexpId = 'ml_res18_IR'
    args.loadstep = loadstep
    args.train_num_steps = loadstep + 30000
    return args


def ml_ir_res18_v1_fit_100(args):
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = v1_rescue_fit_setting(args)
    args = ml_ir_res18_v1_load(args, 1000900)
    args.expId = 'ml_ir_res18_v1_fit_100'
    return args


def ml_ir_res18_v1_fit(args):
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = v1_rescue_fit_setting(args)
    args = ml_ir_res18_v1_load(args, 10009 * 193)
    args.expId = 'ml_ir_res18_v1_fit'
    return args


def ml_ir_res18_v4it_fit(args):
    args = rescue_basic_setting(args)
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = ml_ir_res18_v1_load(args, 10009 * 193)
    args.it_nodes = 'encode_[2:1:11]'
    args.expId = 'ml_ir_res18_v4it_fit'
    args.weight_decay = 1e-2
    return args


def ml3_ir_res18_v1_load(args, loadstep):
    args = ml_load_setting(args)
    args.loadexpId = 'ml3_res18_IR'
    args.loadstep = loadstep
    args.train_num_steps = loadstep + 30000
    return args


def ml3_ir_res18_v1_fit_100(args):
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = v1_rescue_fit_setting(args)
    args = ml3_ir_res18_v1_load(args, 1000900)
    args.expId = 'ml3_ir_res18_v1_fit_100'
    return args


def aug08_ir_res18_v1_load(args, loadstep):
    args.loadport = 27007
    args.loaddbname = 'aug'
    args.loadcolname = 'res18'
    args.v4_nodes = 'encode_[2:1:11]'
    args.loadexpId = 'res18_IR_aug08'
    args.loadstep = loadstep
    args.train_num_steps = loadstep + 30000
    return args


def aug08_ir_res18_v1_fit_100(args):
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = v1_rescue_fit_setting(args)
    args = aug08_ir_res18_v1_load(args, 1000900)
    args.expId = 'aug08_ir_res18_v1_fit_100'
    return args


def rescue_basic_setting(args):
    args = basic_setting(args)
    args.dbname = 'v4it-confirm'
    args.weight_decay = 1e-2
    return args


def ml_ir_res18_v4it_fit_100(args):
    args = rescue_basic_setting(args)
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = ml_ir_res18_v1_load(args, 1000900)
    args.it_nodes = 'encode_[2:1:11]'
    args.expId = 'ml_ir_res18_v4it_fit_100'
    args.weight_decay = 1e-2
    return args


def ir_res18_v4it_fit_100(args):
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = rescue_basic_setting(args)
    args = ir_res18_v1_load(args, 1000900)
    args.expId = 'ir_res18_v4it_fit_100'
    args.weight_decay = 1e-2
    args.it_nodes = 'encode_[2:1:11]'
    return args


def rescue_save_setting(args):
    args.dbname = 'v1-rescue'
    args.nport = 27009
    args.colname = 'pat'
    return args


def pat_ml_load_setting(args):
    args.loadport = 27007
    args.loaddbname = 'multi_loss'
    args.loadcolname = 'res18'
    return args


def pat_ml7_ir_res18_load(args, loadstep):
    args = pat_ml_load_setting(args)
    args.loadexpId = 'ml7_res18_IR'
    args.loadstep = loadstep
    args.train_num_steps = loadstep + 30000
    return args


def pat_inst_stype_setting(args):
    args.inst_model = 'all_spatial'
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'
    return args


def ir_ml7_res18_basic(args):
    args = rescue_save_setting(args)
    args = pat_ml7_ir_res18_load(args, 190 * 10009)
    args = pat_inst_stype_setting(args)

    args.expId = 'res18_ml7_ir'
    return args


def pat_ml5_la_res18_load(args, loadstep):
    args = pat_ml_load_setting(args)
    args.loadexpId = 'ml_res18_LA'
    args.loadstep = loadstep
    args.train_num_steps = loadstep + 30000
    return args


def la_ml5_res18_basic(args):
    args = rescue_save_setting(args)
    args = pat_ml5_la_res18_load(args, 199 * 10009)
    args = pat_inst_stype_setting(args)

    args.expId = 'res18_ml5_la'
    return args


def ir_res18_ecd1_frm_cat_v1_load(args, loadstep):
    args.load_train_setting_func = 'combine_irla_others.res18_encode1_frm_cat'
    args.v4_nodes = 'inst_encode_[1:1:10]'
    args.loadstep = loadstep
    args.train_num_steps = loadstep + 30000
    args = basic_color_bn_setting(args)
    return args


def ir_res18_ecd1_frm_cat_basic(args):
    args = rescue_save_setting(args)
    args = ir_res18_ecd1_frm_cat_v1_load(args, 280 * 10010)
    args.expId = 'ir_res18_ecd1_frm_cat'
    return args


def cate_inst_prep_res18_basic(args):
    args = rescue_save_setting(args)
    args = basic_res18_setting(args)
    args.load_train_setting_func = 'cate_res18_inst_prep'
    args = cate_inst_prep_res18_v1_load(args, 490490)
    args.expId = 'cate_inst_prep_res18'
    return args
