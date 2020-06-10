import nf_exp_settings.shared_settings as shr_sts
import nf_exp_settings.v1_cadena_setting as v1_sts


def res50_basic_setting(args):
    args = shr_sts.basic_setting(args)
    args.dbname = 'res50-fitting'
    args.colname = 'v4it'
    args.weight_decay = 1e-2
    return args


def inst_style_res50(args):
    args = shr_sts.basic_inst_model_setting(args)
    args.inst_res_size = 50
    args.v4_nodes = 'encode_[2:2:19]'
    args.it_nodes = 'encode_[2:2:19]'
    return args


def ir_res50_load(args):
    args.loadcolname = 'control'
    args.loadexpId = 'full50'
    args.loadstep = 260 * 10009
    args.train_num_steps = args.loadstep + 30000
    return args


def ir_res50_v4it_fit(args):
    args = res50_basic_setting(args)
    args = shr_sts.basic_color_bn_setting(args)
    args = inst_style_res50(args)
    args = ir_res50_load(args)

    args.expId = 'ir_1'
    return args


def ir_res50_v4it_fit_bwd(args):
    args = ir_res50_v4it_fit(args)
    args.weight_decay = 1e-1
    args.expId = 'ir_bwd'
    return args


def ir_res50_v4it_fit_swd(args):
    args = ir_res50_v4it_fit(args)
    args.weight_decay = 1e-3
    args.expId = 'ir_swd'
    return args


def res50_v1_basic_setting(args):
    args = v1_sts.v1_cadena_fit_setting(args)
    args.dbname = 'res50-fitting'
    args.colname = 'v1'
    return args


def ir_res50_v1_fit(args):
    args = shr_sts.basic_color_bn_setting(args)
    args = inst_style_res50(args)
    args = ir_res50_load(args)
    args = res50_v1_basic_setting(args)

    args.expId = 'ir'
    return args


def ir_res50_v1_fit_bwd(args):
    args = ir_res50_v1_fit(args)
    args.weight_decay = 1e-1
    args.expId = 'ir_bwd'
    return args


def ir_res50_v1_fit_swd(args):
    args = ir_res50_v1_fit(args)
    args.weight_decay = 1e-3
    args.expId = 'ir_swd'
    return args


def ir_res50_v1_fit_sswd(args):
    args = ir_res50_v1_fit(args)
    args.weight_decay = 1e-4
    args.expId = 'ir_sswd'
    return args


def res50_pat_save_settings(args):
    args.nport = 27009
    args.dbname = 'res50-fitting'
    args.colname = 'pat'
    return args


def res50_cate_shared(args):
    args = shr_sts.basic_res50_setting(args)
    args = res50_pat_save_settings(args)
    args.load_train_setting_func = 'cate_res50'
    return args


def res50_cate_basic_80(args):
    args = res50_cate_shared(args)
    args.expId = 'res50_cate'
    args.loadstep = 80 * 5005
    args.train_num_steps = args.loadstep + 30000
    return args


def res50_cate_even_basic_80(args):
    args = res50_cate_basic_80(args)
    args.expId = 'res50_cate_even'
    args.it_nodes = 'encode_[2:2:18]'
    args.v4_nodes = 'encode_[2:2:18]'
    return args


def ir_res50_odd_basic(args):
    args = res50_pat_save_settings(args)
    args = inst_style_res50(args)
    args = ir_res50_load(args)
    args.loadport = 27009
    args.v4_nodes = 'encode_[3:2:19]'
    args.it_nodes = 'encode_[3:2:19]'
    args.expId = 'res50_ir_odd'
    return args


def ir_res50_basic(args):
    args = res50_pat_save_settings(args)
    args = inst_style_res50(args)
    args = ir_res50_load(args)
    args.loadport = 27009
    args.expId = 'res50_ir'
    return args


def la_res50_shared(args):
    args = res50_pat_save_settings(args)
    args = inst_style_res50(args)
    args.loadport = 27009
    args.loadexpId = 'dyn_km_50_mre10sm'
    return args


def la_res50_basic(args):
    args = la_res50_shared(args)
    args.expId = 'res50_la'
    args.loadstep = 1951755
    args.train_num_steps = args.loadstep + 30000
    return args


def la_res50_odd_basic(args):
    args = la_res50_basic(args)
    args.v4_nodes = 'encode_[3:2:19]'
    args.it_nodes = 'encode_[3:2:19]'
    args.expId = 'res50_la_odd'
    return args


def dc_res50_shared(args):
    args.deepcluster = 'res50_deseq'
    args.it_nodes = 'conv[104:2:121]'
    args.v4_nodes = 'conv[104:2:121]'
    args.loadport = 27009
    return args


def dc_res50_odd_shared(args):
    args.deepcluster = 'res50_deseq'
    args.it_nodes = 'conv[105:2:121]'
    args.v4_nodes = 'conv[105:2:121]'
    args.loadport = 27009
    return args


def dc_res50_v4it_basic(args):
    args = res50_pat_save_settings(args)
    args = dc_res50_shared(args)
    args = shr_sts.basic_setting_wo_db(args)
    args.expId = 'res50_dc_v4it_fx'
    args.train_num_steps = 20000
    return args


def dc_res50_v1_basic(args):
    args = res50_pat_save_settings(args)
    args = dc_res50_shared(args)
    args = v1_sts.v1_fit_wo_db(args)
    args.expId = 'res50_dc_v1_fx'
    args.deepcluster = 'res50_deseq_v1'
    args.train_num_steps = 30000
    return args


def dc_res50_v4it_odd_basic(args):
    args = res50_pat_save_settings(args)
    args = dc_res50_odd_shared(args)
    args = shr_sts.basic_setting_wo_db(args)
    args.expId = 'res50_dc_v4it_odd'
    args.train_num_steps = 20000
    return args


def dc_res50_v1_odd_basic(args):
    args = res50_pat_save_settings(args)
    args = dc_res50_odd_shared(args)
    args = v1_sts.v1_fit_wo_db(args)
    args.expId = 'res50_dc_v1_odd'
    args.deepcluster = 'res50_deseq_v1'
    args.train_num_steps = 30000
    return args


def cmc_res50_shared(args):
    args.deepcluster = 'cmc_res50'
    args.it_nodes = 'conv[0:2:17]'
    args.v4_nodes = 'conv[0:2:17]'
    args.loadport = 27009
    return args


def cmc_res50_only_l_v4it_basic(args):
    args = res50_pat_save_settings(args)
    args = cmc_res50_shared(args)
    args = shr_sts.basic_setting_wo_db(args)
    args.expId = 'res50_cmc_v4it'
    args.deepcluster += ':only_l'
    args.train_num_steps = 20000
    return args


def cmc_res50_only_l_v1_basic(args):
    args = res50_pat_save_settings(args)
    args = cmc_res50_shared(args)
    args = v1_sts.v1_fit_wo_db(args)
    args.expId = 'res50_cmc_v1'
    args.deepcluster = 'cmc_res50_v1'
    args.deepcluster += ':only_l'
    args.train_num_steps = 30000
    return args


def cmc_res50_odd_shared(args):
    args.deepcluster = 'cmc_res50'
    args.it_nodes = 'conv[1:2:17]'
    args.v4_nodes = 'conv[1:2:17]'
    args.loadport = 27009
    return args


def cmc_res50_only_l_odd_v4it_basic(args):
    args = res50_pat_save_settings(args)
    args = cmc_res50_odd_shared(args)
    args = shr_sts.basic_setting_wo_db(args)
    args.expId = 'res50_cmc_odd_v4it'
    args.deepcluster += ':only_l'
    args.train_num_steps = 20000
    return args


def cmc_res50_only_l_odd_v1_basic(args):
    args = res50_pat_save_settings(args)
    args = cmc_res50_odd_shared(args)
    args = v1_sts.v1_fit_wo_db(args)
    args.expId = 'res50_cmc_odd_v1'
    args.deepcluster = 'cmc_res50_v1'
    args.deepcluster += ':only_l'
    args.train_num_steps = 30000
    return args


def res50_untrn_shared(args):
    args = shr_sts.basic_res50_setting(args)
    args = res50_pat_save_settings(args)
    args.network_func = 'get_resnet_50'
    args.loadport = 27009
    args.loaddbname = 'res50-fitting'
    args.loadcolname = 'pat'
    return args


def res50_untrn_basic(args):
    args = res50_untrn_shared(args)
    args.expId = 'res50_untrn'
    args.train_num_steps = 30000
    return args


def res50_untrn_even_basic(args):
    args = res50_untrn_shared(args)
    args.expId = 'res50_untrn_even'
    args.train_num_steps = 30000
    args.it_nodes = 'encode_[2:2:18]'
    args.v4_nodes = 'encode_[2:2:18]'
    return args
