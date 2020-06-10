import nf_exp_settings.shared_settings as shared_settings
import nf_exp_settings.v1_cadena_setting as v1_cadena_setting
import nf_exp_settings.jrnl_setting as jrnl_setting
import nf_exp_settings.v4it_hvm_setting as v4it_hvm_setting


def pat_basic_save_setting(args):
    args.dbname = 'pat_save'
    args.nport = 27009
    args.colname = 'gnrl'
    return args


def la_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args = shared_settings.basic_inst_model_setting(args)
    args = v1_cadena_setting.la_res18_v1_load(args)
    args.loadport = 27009
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'
    args.expId = 'la_res18'
    return args


def la_sobel_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args = shared_settings.basic_inst_model_setting(args)
    args.loadport = 27009
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'
    args.expId = 'la_sobel_res18'
    args.ckpt_file = '/mnt/fs6/honglinc/'\
            + 'tf_experiments/models/la_multi_modal/'\
            + 'test/res18_LA_Sobel/model.ckpt-2001800'
    args.input_mode = 'sobel'
    args.train_num_steps = 2001800 + 30000
    return args


def la_gray_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args = shared_settings.basic_inst_model_setting(args)
    args.loadport = 27009
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'
    args.expId = 'la_gray_res18'
    args.ckpt_file = '/mnt/fs6/honglinc/'\
            + 'tf_experiments/models/la_multi_modal/'\
            + 'test/res18_LA_Gray/model.ckpt-2001800'
    args.input_mode = 'gray'
    args.train_num_steps = 2001800 + 30000
    return args


def la_L_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args = shared_settings.basic_inst_model_setting(args)
    args.loadport = 27009
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'
    args.expId = 'la_L_res18'
    args.ckpt_file = '/mnt/fs6/honglinc/'\
            + 'tf_experiments/models/la_multi_modal/'\
            + 'test/res18_LA_L/model.ckpt-2001800'
    args.input_mode = 'L'
    args.train_num_steps = 2001800 + 30000
    return args


def ir_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args = shared_settings.basic_inst_model_setting(args)
    args = v1_cadena_setting.ir_res18_v1_load(args)
    args.loadport = 27009
    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'
    args.expId = 'ir_res18'
    return args


def ir_vm_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args = jrnl_setting.ir_vm_res18_load(args)
    args.expId = 'ir_vm_res18'
    return args


def la_cate_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args.load_train_setting_func = 'combine_irla_others.res18_la_cate'
    args.loadstep = 160 * 10010
    args.train_num_steps = args.loadstep + 30000
    args.expId = 'la_cate_res18'
    return args


def la_cate_early_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args.load_train_setting_func = 'combine_irla_others.res18_la_cate_early'
    args.loadstep = 150 * 10010
    args.train_num_steps = args.loadstep + 30000
    args.expId = 'la_cate_early_res18'
    return args


def la_cate_even_early_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args.load_train_setting_func = 'combine_irla_others.res18_la_cate_even_early'
    args.loadstep = 160 * 10010
    args.train_num_steps = args.loadstep + 30000
    args.expId = 'la_cate_even_early_res18'
    return args


def la_cate_early_enc6_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args.load_train_setting_func = 'combine_irla_others.res18_la_cate_early_enc6'
    args.loadstep = 110 * 10010
    args.train_num_steps = args.loadstep + 30000
    args.expId = 'la_cate_early_enc6_res18'
    return args


def cate_inst_prep_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args.load_train_setting_func = 'cate_res18_inst_prep'
    args.loadstep = 490490
    args.train_num_steps = args.loadstep + 30000
    args.expId = 'cate_inst_prep_res18'
    return args


def cate_sobel_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args.load_train_setting_func = 'cate_sobel_res18'
    args.loadstep = 490490
    args.train_num_steps = args.loadstep + 30000
    args.expId = 'cate_sobel_res18'
    return args


def ir_res18_encode1to3_frm_cate_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args.v4_nodes = 'inst_encode_[1:1:10]'
    args.it_nodes = 'inst_encode_[1:1:10]'
    args.load_train_setting_func = 'combine_irla_others.res18_encode1to3_frm_cate'
    args.loadstep = 250 * 10010
    args.train_num_steps = args.loadstep + 30000
    args.expId = 'ir_res18_1to3_frm_cate'
    return args


def ir_res18_encode1to3_frm_ir_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args.v4_nodes = 'inst_encode_[1:1:10]'
    args.it_nodes = 'inst_encode_[1:1:10]'
    args.load_train_setting_func = 'combine_irla_others.res18_encode1to3_frm_ir'
    args.loadstep = 250 * 10010
    args.train_num_steps = args.loadstep + 30000
    args.expId = 'ir_res18_1to3_frm_ir'
    return args


def cate_seed0_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args.load_train_setting_func = 'cate_res18_exp0'
    args.loadstep = 490490
    args.train_num_steps = args.loadstep + 30000
    args.expId = 'cate_seed0_res18'
    #args.it_nodes = 'encode_1'
    #args.v4_nodes = None
    #args.v4_nodes = 'encode_1'
    #args.it_nodes = None
    return args


def cate_seed1_res18_basic(args):
    args = cate_seed0_res18_basic(args)
    args.load_train_setting_func = 'cate_res18_exp1'
    args.expId = 'cate_seed1_res18'
    return args


def cate_seed2_res18_basic(args):
    args = cate_seed0_res18_basic(args)
    args.load_train_setting_func = 'cate_res18_exp2'
    args.expId = 'cate_seed2_res18'
    return args


def cate_seed3_res18_basic(args):
    args = cate_seed0_res18_basic(args)
    args.load_train_setting_func = 'cate_res18_exp3'
    args.expId = 'cate_seed3_res18'
    return args


def cmc_res18_shared(args):
    args.deepcluster = 'cmc_res18'
    args.it_nodes = 'conv[0:1:9]'
    args.v4_nodes = 'conv[0:1:9]'
    args.loadport = 27009
    args.train_num_steps = 30000
    return args


def cmc_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = cmc_res18_shared(args)
    args.expId = 'cmc_res18'
    return args


def cmc_res18_v1_basic(args):
    args = pat_basic_save_setting(args)
    args = cmc_res18_shared(args)
    args.expId = 'cmc_res18_v1'
    args.deepcluster = 'cmc_res18_v1'
    return args


def la_cmc_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = cmc_res18_shared(args)
    args.deepcluster = 'la_cmc_res18'
    args.expId = 'la_cmc_res18'
    return args


def la_cmc_res18_v1_basic(args):
    args = pat_basic_save_setting(args)
    args = cmc_res18_shared(args)
    args.expId = 'la_cmc_res18_v1'
    args.deepcluster = 'la_cmc_res18_v1'
    return args


def la_cmc_res18v1_v1_basic(args):
    args = pat_basic_save_setting(args)
    args = cmc_res18_shared(args)
    args.expId = 'la_cmc_res18v1_v1'
    args.deepcluster = 'la_cmc_res18v1_v1'
    return args


def pt_official_res18_basic(args):
    args = pat_basic_save_setting(args)
    args = cmc_res18_shared(args)
    args.deepcluster = 'pt_official_res18'
    args.expId = 'pt_official_res18'
    return args


def pt_official_res18_v1_basic(args):
    args = pat_basic_save_setting(args)
    args = cmc_res18_shared(args)
    args.expId = 'pt_official_res18_v1'
    args.deepcluster = 'pt_official_res18_v1'
    return args


def pt_official_res18_var6_basic(args):
    args = pat_basic_save_setting(args)
    args = cmc_res18_shared(args)
    args.deepcluster = 'pt_official_res18_var6'
    args.expId = 'pt_official_res18'
    return args


def pat_basic_rep_save_setting(args):
    args.dbname = 'pat_save'
    args.nport = 27009
    args.colname = 'gnrl_rep'
    return args


def la_vm_res18_basic(args):
    args = pat_basic_rep_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args.load_train_setting_func = 'combine_irla_others.res18_la'
    args.loadstep = 250 * 10010
    args.train_num_steps = args.loadstep + 30000
    args.expId = 'la_vm_res18'
    return args


def la_vm_s1_res18_basic(args):
    args = la_vm_res18_basic(args)
    args.load_train_setting_func = 'combine_irla_others.res18_la_s1'
    args.expId = 'la_vm_s1_res18'
    return args


def la_vm_s2_res18_basic(args):
    args = la_vm_res18_basic(args)
    args.load_train_setting_func = 'combine_irla_others.res18_la_s2'
    args.expId = 'la_vm_s2_res18'
    return args


def ir_vm_new_res18_basic(args):
    args = pat_basic_rep_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args.load_train_setting_func = 'combine_irla_others.res18_ir'
    args.loadstep = 250 * 10010
    args.train_num_steps = args.loadstep + 30000
    args.expId = 'ir_vm_new_res18'
    return args


def ir_vm_s1_new_res18_basic(args):
    args = ir_vm_new_res18_basic(args)
    args.load_train_setting_func = 'combine_irla_others.res18_ir_s1'
    args.expId = 'ir_vm_s1_new_res18'
    return args


def ir_vm_s2_new_res18_basic(args):
    args = ir_vm_new_res18_basic(args)
    args.load_train_setting_func = 'combine_irla_others.res18_ir_s2'
    args.expId = 'ir_vm_s2_new_res18'
    return args


def la_new_res18_basic(args):
    args = la_res18_basic(args)
    args = pat_basic_rep_save_setting(args)
    args.expId = 'la_new_res18'
    args.ckpt_file = '/mnt/fs6/honglinc/'\
            + 'tf_experiments/models/la_multi_modal/'\
            + 'test/res18_LA_RGB/model.ckpt-2001800'
    args.train_num_steps = 2001800 + 30000
    return args


def pat_basic_tpu_rep_save_setting(args):
    args.dbname = 'tpu_pat_save'
    args.nport = 27009
    args.colname = 'gnrl_rep'
    args.loadport = 27009
    return args


def rp_res18_s1_basic(args):
    args = pat_basic_tpu_rep_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args = v4it_hvm_setting.rp_res18_load(args)
    step_num = 1080972
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/rp_res18_s1/model.ckpt-%i' % step_num
    args.train_num_steps = step_num + 30000
    args.expId = 'rp_res18_s1'
    return args


def rp_res18_s2_basic(args):
    args = pat_basic_tpu_rep_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args = v4it_hvm_setting.rp_res18_load(args)
    step_num = 1020918
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/rp_res18_s2/model.ckpt-%i' % step_num
    args.train_num_steps = step_num + 30000
    args.expId = 'rp_res18_s2'
    return args


def col_res18_s1_basic(args):
    args = pat_basic_tpu_rep_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args = v4it_hvm_setting.col_res18_load(args)
    step_num = 4303870
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18_s1/model.ckpt-%i' % step_num
    args.train_num_steps = step_num + 30000
    args.expId = 'col_res18_s1'
    return args


def col_res18_s2_basic(args):
    args = pat_basic_tpu_rep_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args = v4it_hvm_setting.col_res18_load(args)
    step_num = 5184662
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/col_res18_s2/model.ckpt-%i' % step_num
    args.train_num_steps = step_num + 30000
    args.expId = 'col_res18_s2'
    return args


def depth_res18_s1_basic(args):
    args = pat_basic_tpu_rep_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args = v4it_hvm_setting.depth_res18_load(args)
    step_num = 3583222
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/depth_res18_s1/model.ckpt-%i' % step_num
    args.train_num_steps = step_num + 30000
    args.expId = 'depth_res18_s1'
    return args


def depth_res18_s2_basic(args):
    args = pat_basic_tpu_rep_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args = v4it_hvm_setting.depth_res18_load(args)
    step_num = 3102790
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/depth_res18_s2/model.ckpt-%i' % step_num
    args.train_num_steps = step_num + 30000
    args.expId = 'depth_res18_s2'
    return args


def ae_res18_basic(args):
    args = pat_basic_rep_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args.load_train_setting_func = 'other_tasks.res18_AE_imagenet'
    args.loadstep = 130 * 10010
    args.train_num_steps = args.loadstep + 30000
    args.expId = 'ae_res18'
    return args


def ae_res18_s1_basic(args):
    args = ae_res18_basic(args)
    args.load_train_setting_func = 'other_tasks.res18_AE_imagenet_seed1'
    args.expId = 'ae_res18_s1'
    return args


def ae_res18_s2_basic(args):
    args = ae_res18_basic(args)
    args.load_train_setting_func = 'other_tasks.res18_AE_imagenet_seed2'
    args.expId = 'ae_res18_s2'
    return args


def cpc_res18_basic(args):
    args = pat_basic_rep_save_setting(args)
    args = shared_settings.basic_res18_setting(args)
    args.load_train_setting_func = 'other_tasks.res18_cpc_imagenet_tpu'
    args.loadstep = 130 * 10010
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/cpc/model.ckpt-%i' % args.loadstep
    args.train_num_steps = args.loadstep + 30000
    args.expId = 'cpc_res18'
    return args


def cpc_res18_s1_basic(args):
    args = cpc_res18_basic(args)
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/cpc_seed1/model.ckpt-%i' % (130*10010)
    args.expId = 'cpc_res18_s1'
    return args


def cpc_res18_s2_basic(args):
    args = cpc_res18_basic(args)
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/cpc_seed2/model.ckpt-%i' % (130*10010)
    args.expId = 'cpc_res18_s2'
    return args
