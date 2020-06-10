import exp_settings.shared_settings as shared_sts
from tensorflow.python import pywrap_tensorflow
import json


def save_setting(args):
    args.nport = 27007
    args.dbname = 'irla_and_others'
    args.collname = 'res18'
    return args


def basic_inst_setting(args):
    args = shared_sts.basic_inst_setting(args)
    args.tpu_flag = 1
    args.whichopt = 0
    args.whichimagenet = 'full_widx'
    args.init_type = 'variance_scaling_initializer'
    return args


def cmb_rp_inst_setting(args):
    args = basic_inst_setting(args)
    args.network_func = 'combine_irla_others.get_rp_inst_resnet_18'
    args.dataconfig = 'dataset_config_imn_rpimn.json'
    return args


def res18_cmb_rp_inst(args):
    args = shared_sts.basic_setting(args)
    args = save_setting(args)
    args = shared_sts.bs_64_setting(args)
    args = cmb_rp_inst_setting(args)

    args.expId = 'cmb_rp_inst'
    args.lr_boundaries = '2500011'
    return args


def res18_cmb_rp7_inst(args):
    args = res18_cmb_rp_inst(args)
    args.network_func_kwargs = '{"rp_layer_offset": 2}'
    args.expId = 'cmb_rp7_inst'
    args.lr_boundaries = '2500011'
    return args


def res18_cmb_rp7_inst_bn(args):
    args = res18_cmb_rp_inst(args)
    args.network_func_kwargs = '{"rp_layer_offset": 2}'
    args.expId = 'cmb_rp7_inst_bn'
    args.lr_boundaries = None
    args.ignorebname_new = 0
    return args


def load_from_inst_prep_res18_cate(args):
    args.loadport = 27007
    args.load_dbname = 'cate_aug'
    args.load_collname = 'res18'
    args.loadexpId = 'inst_prep'
    return args


def res18_encode1_frm_cat(args):
    args = shared_sts.basic_setting(args)
    args = basic_inst_setting(args)

    args = save_setting(args)
    args = shared_sts.bs_128_less_save_setting(args)

    args.dataconfig = 'dataset_config_image.cfg'
    args.expId = 'encode1_frm_cat'
    args = load_from_inst_prep_res18_cate(args)
    args.network_func = 'combine_irla_others.get_resnet_18_fx_encd1'
    # Uncomment for beginning
    #args.load_param_dict = '{"encode1/weights": "inst_encode1/weights", "encode1/bias": "inst_encode1/bias"}'
    args.lr_boundaries = '1780011,2400011'
    return args


def get_fx_encode1toX_param_dict_from_cate_model(no_layers_fixed=3):
    path_to_model = '/mnt/fs4/chengxuz/tfutils_ckpts/cate_aug/res18/inst_prep/checkpoint-490490'
    reader = pywrap_tensorflow.NewCheckpointReader(path_to_model)
    load_param_dict = {}
    for each_key in reader.get_variable_to_shape_map():
        need_flag = False
        for idx_layer in range(no_layers_fixed + 1):
            if each_key.startswith('encode%i/' % idx_layer) \
                    and (not each_key.endswith('/Momentum')):
                need_flag = True
        if need_flag:
            load_param_dict[each_key] = "inst_" + each_key
    return json.dumps(load_param_dict)


def res18_encode1to3_frm_cate(args):
    args = shared_sts.basic_setting(args)
    args = basic_inst_setting(args)

    args = save_setting(args)
    args = shared_sts.bs_128_less_save_setting(args)
    args.dataconfig = 'dataset_config_image.cfg'
    args.expId = 'encode1to3_frm_cate_fx'
    args = load_from_inst_prep_res18_cate(args)
    args.network_func = 'combine_irla_others.get_resnet_18_fx_encdX'
    # Uncomment for beginning
    args.load_param_dict = get_fx_encode1toX_param_dict_from_cate_model()
    args.lr_boundaries = '1780011,2400011'
    return args


def load_from_inst_prep_res18_ir(args):
    args.loadport = 27006
    args.load_dbname = 'combinet-test'
    args.load_collname = 'combinet'
    args.loadexpId = 'inst_res18_newp_dset'
    return args


def res18_encode1to3_frm_ir(args):
    args = shared_sts.basic_setting(args)
    args = basic_inst_setting(args)

    args = save_setting(args)
    args = shared_sts.bs_128_less_save_setting(args)
    args.dataconfig = 'dataset_config_image.cfg'
    args.expId = 'encode1to3_frm_ir'
    args = load_from_inst_prep_res18_ir(args)
    args.network_func = 'combine_irla_others.get_resnet_18_fx_encdX'
    # Uncomment for beginning
    args.load_param_dict = get_fx_encode1toX_param_dict_from_cate_model()
    args.lr_boundaries = '1780011,2400011'
    return args


def res18_ir(args):
    args = shared_sts.basic_setting(args)
    args = basic_inst_setting(args)

    args = save_setting(args)
    args = shared_sts.bs_128_less_save_setting(args)
    args.network_func = 'get_resnet_18'
    args.network_func_kwargs = '{"num_cat": 128}'
    args.dataconfig = 'dataset_config_image.cfg'
    args.expId = 'ir'
    args.lr_boundaries = '1780011,2400011'
    return args


def res18_ir_s1(args):
    args = res18_ir(args)
    args.expId = 'ir_s1'
    args.seed = 1
    return args


def res18_ir_s2(args):
    args = res18_ir(args)
    args.expId = 'ir_s2'
    args.seed = 2
    return args


def res18_ir_saycam(args):
    args = res18_ir(args)
    args.dataconfig = 'dataset_task_config_saycam_ir.json'
    args.expId = 'ir_saycam'
    args.nport = 26001
    args.validation_skip = 1
    args.network_func = 'get_resnet_18_saycam'
    return args


def res18_ir_saycam_all_125(args):
    args = res18_ir(args)
    args.dataconfig = 'dataset_task_config_saycam_all_ir.json'
    args.expId = 'ir_saycam_all_125'
    args.nport = 26001
    args.validation_skip = 1
    args.network_func = 'get_resnet_18_saycam'
    args.instance_data_len = 26894250 // 125
    return args


def res18_ir_saycam_all_25(args):
    args = res18_ir(args)
    args.dataconfig = 'dataset_task_config_saycam_all25_ir.json'
    args.expId = 'ir_saycam_all_25'
    args.nport = 26001
    args.validation_skip = 1
    args.network_func = 'get_resnet_18_saycam'
    args.instance_data_len = 26894250 // 25
    return args


def res18_ir_depth(args):
    args = shared_sts.basic_setting(args)
    args = basic_inst_setting(args)

    args = save_setting(args)
    args = shared_sts.bs_64_less_save_setting(args)
    args.network_func = 'combine_irla_others.get_resnet18_inst_and_depth'
    args.dataconfig = 'dataset_config_pbrimage.cfg'
    args.expId = 'ir_depth'
    args.lr_boundaries = '1300011'
    return args


def res18_ir_depth_7(args):
    args = res18_ir_depth(args)
    args.network_func_kwargs = '{"which_encode_to_depth": 7}'
    args.expId = 'ir_depth_7'
    return args


def res18_ir_depth_7_bn(args):
    args = res18_ir_depth(args)
    args.network_func_kwargs = '{"which_encode_to_depth": 7}'
    args.expId = 'ir_depth_7_bn_cont'
    args.ignorebname_new = 0
    args.lr_boundaries = None
    args.loadexpId = 'ir_depth_7_bn'
    args.loadstep = 120 * 10010
    return args


def basic_la_setting(args):
    args = basic_inst_setting(args)
    args.imgnt_w_idx = True
    args.instance_task = False
    args.network_func = 'get_resnet_18'
    args.network_func_kwargs = '{"num_cat": 128}'
    args.with_rep = 1
    return args


def get_load_param_dict_ir2la(dataset_name='imagenet'):
    mb_prev_name = dataset_name + "/memory_bank"
    mb_now_name = dataset_name + "_LA/memory_bank"
    lb_prev_name = dataset_name + "/all_labels"
    lb_now_name = dataset_name + "_LA/all_labels"
    return json.dumps(
            {mb_prev_name: mb_now_name,
             lb_prev_name: lb_now_name})


def res18_la(args):
    args = shared_sts.basic_setting(args)
    args = basic_la_setting(args)

    args = save_setting(args)
    args = shared_sts.bs_128_less_save_setting(args)
    args.dataconfig = 'dataset_task_config_image_la.json'
    args = load_from_inst_prep_res18_ir(args)
    args.loadstep = 100000
    args.expId = 'la'
    # Uncomment for beginning
    #args.load_param_dict = get_load_param_dict_ir2la()
    args.lr_boundaries = '1740011,2400011'
    return args


def load_from_res18_ir_s1(args):
    args.loadport = 27007
    args.load_dbname = 'irla_and_others'
    args.load_collname = 'res18'
    args.loadexpId = 'ir_s1'
    args.loadstep = 100100
    return args


def res18_la_s1(args):
    args = res18_la(args)
    args = load_from_res18_ir_s1(args)
    args.expId = 'la_s1'
    args.seed = 1
    #args.load_param_dict = get_load_param_dict_ir2la()
    return args


def load_from_res18_ir_s2(args):
    args = load_from_res18_ir_s1(args)
    args.loadexpId = 'ir_s2'
    return args


def res18_la_s2(args):
    args = res18_la(args)
    args = load_from_res18_ir_s2(args)
    args.expId = 'la_s2'
    args.seed = 2
    #args.load_param_dict = get_load_param_dict_ir2la()
    return args


def load_from_res18_ir_saycam(args):
    args.loadport = 26001
    args.load_dbname = 'irla_and_others'
    args.load_collname = 'res18'
    args.loadexpId = 'ir_saycam'
    args.loadstep = 100100
    return args


def res18_la_saycam(args):
    args = res18_la(args)
    args = load_from_res18_ir_saycam(args)
    args.dataconfig = 'dataset_task_config_saycam_la.json'
    args.expId = 'la_saycam'
    args.nport = 26001
    args.validation_skip = 1
    args.network_func = 'get_resnet_18_saycam'

    # Uncomment for beginning
    #args.load_param_dict = get_load_param_dict_ir2la('saycam')
    return args


def res18_la_saycam_all_125(args):
    args = res18_la(args)
    args = load_from_res18_ir_saycam(args)
    args.dataconfig = 'dataset_task_config_saycam_all_la.json'
    args.expId = 'la_saycam_all_125_2'
    args.nport = 26001
    args.validation_skip = 1
    args.network_func = 'get_resnet_18_saycam'
    args.loadexpId = 'ir_saycam_all_125'
    args.instance_data_len = 26894250 // 125

    # Uncomment for beginning
    args.load_param_dict = get_load_param_dict_ir2la('saycam')
    return args


def res18_la_saycam_all_25(args):
    args = res18_la(args)
    args = load_from_res18_ir_saycam(args)
    args.dataconfig = 'dataset_task_config_saycam_all25_la.json'
    args.expId = 'la_saycam_all_25_2'
    args.nport = 26001
    args.validation_skip = 1
    args.network_func = 'get_resnet_18_saycam'
    args.loadexpId = 'ir_saycam_all_25'
    args.instance_data_len = 26894250 // 25

    # Uncomment for beginning
    args.load_param_dict = get_load_param_dict_ir2la('saycam')
    return args


def basic_la_cate_setting(args):
    args = basic_inst_setting(args)
    args.imgnt_w_idx = True
    args.instance_task = False
    args.network_func = 'get_resnet_18_inst_and_cate'
    args.with_rep = 1
    return args


def load_from_res18_ir_cate(args):
    args.loadport = 27009
    args.load_dbname = 'combine_instance'
    args.load_collname = 'combinet'
    args.loadexpId = 'res18_inst_cate_1'
    return args


def res18_la_cate(args):
    args = shared_sts.basic_setting(args)
    args = basic_la_cate_setting(args)

    args = save_setting(args)
    args = shared_sts.bs_256_less_save_setting(args)
    args.dataconfig = 'dataset_task_config_image_la_cate.json'
    args = load_from_res18_ir_cate(args)
    args.loadstep = 50050
    args.expId = 'la_cate'
    # Uncomment for beginning
    #args.load_param_dict = get_load_param_dict_ir2la()
    args.lr_boundaries = '1040011,1550011'
    return args


def res18_la_cate_early(args):
    args = res18_la_cate(args)
    args.network_func = 'get_resnet_18_inst_and_cate_early_memory'
    args.loadexpId = 'res18_inst_cate_early'
    args.expId = 'la_cate_early'
    # Uncomment for beginning
    #args.load_param_dict = get_load_param_dict_ir2la()
    args.lr_boundaries = '940011,1380011'
    return args


def res18_la_cate_even_early(args):
    args = res18_la_cate(args)
    args.network_func = 'get_resnet_18_inst_and_cate_even_early_memory'
    args.loadexpId = 'res18_inst_cate_even_early'
    args.expId = 'la_cate_even_early'
    # Uncomment for beginning
    #args.load_param_dict = get_load_param_dict_ir2la()
    args.lr_boundaries = '1000011,1510011'
    return args


def res18_la_cate_early_enc6(args):
    args = res18_la_cate(args)
    args.network_func = 'get_resnet_18_inst_and_cate_memory_enc6'
    args.loadexpId = 'res18_inst_cate_early_enc6'
    args.expId = 'la_cate_early_enc6'
    # Uncomment for beginning
    #args.load_param_dict = get_load_param_dict_ir2la()
    args.lr_boundaries = '760011,1010011'
    return args


def res18_ir_cate_separate_bn(args):
    args = shared_sts.basic_setting(args)
    args = basic_la_cate_setting(args)

    args = save_setting(args)
    args = shared_sts.bs_256_less_save_setting(args)
    args.network_func = 'get_resnet_18_cate_inst_branch2'
    args.dataconfig = 'dataset_task_config_image_cate_ir_branch2.json'
    args.expId = 'ir_cate_bn'
    args.lr_boundaries = "500000,800000,960000"
    args.ignorebname_new = 0
    return args
