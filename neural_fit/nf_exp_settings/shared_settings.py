def basic_setting_wo_db(args):
    args.whichopt = 1
    args.weight_decay = 1e-3
    args.batchsize = 64
    args.use_dataset_inter = True
    args.fre_valid = 160
    args.fre_metric = 160
    return args


def var6_basic_setting_wo_db(args):
    args.whichopt = 1
    args.weight_decay = 1e-3
    args.batchsize = 64
    args.use_dataset_inter = True
    args.fre_valid = 160
    args.fre_metric = 160
    args.dataset_type = 'hvm_var6'
    return args


def basic_setting(args):
    args = basic_setting_wo_db(args)
    args.loadport = 27009
    args.nport = 27009
    args.dbname = 'new-neuralfit'
    return args


def basic_color_bn_setting(args):
    args.rp_sub_mean = 1
    args.div_std = 1
    args.tpu_flag = 1
    return args


def basic_res18_setting(args):
    args = basic_color_bn_setting(args)

    args.it_nodes = 'encode_[1:1:10]'
    args.v4_nodes = 'encode_[1:1:10]'
    return args


def basic_res50_setting(args):
    args = basic_color_bn_setting(args)

    args.it_nodes = 'encode_[1:2:18]'
    args.v4_nodes = 'encode_[1:2:18]'
    return args


def basic_mt_res18_setting(args):
    args = basic_color_bn_setting(args)

    args.it_nodes = 'encode_[1:1:10],ema_encode_[1:1:10]'
    args.v4_nodes = 'encode_[1:1:10],ema_encode_[1:1:10]'
    args.mean_teacher = 1
    return args


def basic_inst_model_setting(args):
    args.inst_model = 'all_spatial'
    args.loaddbname = 'instance_task'
    args.loadcolname = 'dynamic_clustering'
    return args


def basic_convrnn_model_setting(args):
    args.convrnn_model = True
    args.it_nodes = 'conv[0:1:8]'
    args.v4_nodes = 'conv[0:1:8]'
    return args


def convrnn_pretrain_setting(args):
    args.loadport = 27021
    args.loaddbname = 'integ'
    args.loadcolname = 'enet07L'
    args.loadexpId = 'unr0_bs640_0'
    return args


def bs_basic_setting(args):
    args.loadport = 27009
    args.nport = 27009
    args.batchsize = 64
    args.fre_valid = 160
    args.fre_metric = 160
    args.dbname = 'bs-neuralfit'
    return args


def bs_res18_setting(args):
    args.model_type = 'vm_model'
    args.prep_type = 'mean_std'
    args.setting_name = 'part3_cate_res18'
    return args


def deepcluster_res18_deseq(args):
    args.deepcluster = 'res18_deseq'
    args.train_num_steps = 30000
    args.it_nodes = 'conv[104:1:113]'
    args.v4_nodes = 'conv[104:1:113]'
    return args
