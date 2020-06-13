import exp_settings.shared_settings as shared_sts
from exp_settings.ir import basic_inst_setting


def save_setting(args):
    args.nport = 27007
    args.dbname = 'pub_la'
    args.collname = 'res18'
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


def load_from_res18_ir_s0(args):
    args.loadport = 27007
    args.load_dbname = 'pub_ir'
    args.load_collname = 'res18'
    args.loadexpId = 'ir_s0'
    args.loadstep = 100100
    return args


def res18_la_s0(args):
    args = shared_sts.basic_setting(args)
    args = basic_la_setting(args)

    args = save_setting(args)
    args = shared_sts.bs_128_less_save_setting(args)
    args.dataconfig = 'dataset_task_config_image_la.json'
    args = load_from_res18_ir_s0(args)
    args.expId = 'la_s0'
    # Uncomment for beginning
    args.load_param_dict = get_load_param_dict_ir2la()
    args.lr_boundaries = '1740011,2400011'
    args.train_num_steps = 2610011
    return args


def load_from_res18_ir_s1(args):
    args = load_from_res18_ir_s0(args)
    args.loadexpId = 'ir_s1'
    return args


def res18_la_s1(args):
    args = res18_la(args)
    args = load_from_res18_ir_s1(args)
    args.expId = 'la_s1'
    args.seed = 1
    args.load_param_dict = get_load_param_dict_ir2la()
    return args


def load_from_res18_ir_s2(args):
    args = load_from_res18_ir_s0(args)
    args.loadexpId = 'ir_s2'
    return args


def res18_la_s2(args):
    args = res18_la(args)
    args = load_from_res18_ir_s2(args)
    args.expId = 'la_s2'
    args.seed = 2
    args.load_param_dict = get_load_param_dict_ir2la()
    return args
