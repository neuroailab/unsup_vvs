from nf_exp_settings.shared_settings import basic_res18_setting, \
        basic_inst_model_setting, \
        convrnn_pretrain_setting, basic_convrnn_model_setting


def v1_cadena_basic_setting(args):
    args.loadport = 27009
    args.nport = 27009
    args.batchsize = 60
    args.use_dataset_inter = True
    args.dbname = 'new-neuralfit-v1'
    args.objectome_zip = True
    args.gen_features = 1
    args.image_prefix = '/mnt/fs4/chengxuz/v1_cadena_related/images'
    args.obj_dataset_type = 'v1_cadena'
    return args


def cate_res18_v1_cadena(args):
    args = v1_cadena_basic_setting(args)
    args = basic_res18_setting(args)

    args.load_train_setting_func = 'part3_cate_res18'
    step_num = 64 * 10009
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/res18/model.ckpt-%i' % step_num
    args.expId = 'cate_res18_v1_cadena'
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v1_cadena_related/results/cate.hdf5'
    return args


def la_res18_v1_cadena(args):
    args = v1_cadena_basic_setting(args)
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)

    args.it_nodes = 'encode_[2:1:11]'
    args.v4_nodes = 'encode_[2:1:11]'
    args.expId = 'la_res18_v1_cadena'
    args.loadexpId = 'dyn_km_or_mre10_sm'
    args.loadstep = 1931737
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v1_cadena_related/results/la_res18.hdf5'
    return args


def convrnn_v1_cadena(args):
    args = v1_cadena_basic_setting(args)
    args = basic_res18_setting(args)
    args = convrnn_pretrain_setting(args)
    args = basic_convrnn_model_setting(args)

    args.expId = 'convrnn_v1_cadena'
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v1_cadena_related/results/convrnn.hdf5'
    return args


def convrnn_v1_cadena_im46(args):
    args = v1_cadena_basic_setting(args)
    args = basic_res18_setting(args)
    args = convrnn_pretrain_setting(args)
    args = basic_convrnn_model_setting(args)

    args.expId = 'convrnn_v1_cadena_im46'
    args.gen_hdf5path = '/mnt/fs4/chengxuz/v1_cadena_related/results/convrnn_im46.hdf5'
    args.cadena_im_size = 46
    return args


def v1_fit_wo_db(args):
    args.batchsize = 64
    args.whichopt = 1
    args.weight_decay = 1e-2
    args.batchsize = 64
    args.use_dataset_inter = True
    args.fre_valid = 300
    args.fre_metric = 300
    args.dataset_type = 'v1_cadena'
    args.v1v2_folder = '/mnt/fs4/chengxuz/v1_cadena_related/tfrs'
    args.img_out_size = 40
    args.it_nodes = None
    return args


def v1_cadena_fit_setting(args):
    args = v1_fit_wo_db(args)
    args.loadport = 27009
    args.nport = 27009
    args.dbname = 'new-neuralfit-v1'
    return args


def cate_res18_v1_load(args):
    args.load_train_setting_func = 'part3_cate_res18'
    step_num = 64 * 10009
    args.ckpt_file = '/mnt/fs4/chengxuz/tpu_ckpts/res18/model.ckpt-%i' % step_num
    args.train_num_steps = step_num + 30000
    return args


def cate_res18_v1_fit(args):
    args = basic_res18_setting(args)
    args = v1_cadena_fit_setting(args)
    args = cate_res18_v1_load(args)

    args.expId = 'cate_res18_v1_fit'
    return args


def cate_res18_v1_fit_bwd(args):
    args = cate_res18_v1_fit(args)

    args.weight_decay = 1e-1
    args.expId = 'cate_res18_v1_fit_bwd'
    return args


def cate_res18_v1_fit_swd(args):
    args = cate_res18_v1_fit(args)

    args.weight_decay = 1e-3
    args.expId = 'cate_res18_v1_fit_swd'
    return args


def la_res18_v1_load(args, loadstep=1931737):
    args.v4_nodes = 'encode_[2:1:11]'
    args.loadexpId = 'dyn_km_or_mre10_sm'
    args.loadstep = loadstep
    args.train_num_steps = loadstep + 30000
    return args


def la_res18_v1_fit(args):
    args = basic_res18_setting(args)
    args = v1_cadena_fit_setting(args)
    args = basic_inst_model_setting(args)
    args = la_res18_v1_load(args)

    args.expId = 'la_res18_v1_fit'
    return args


def la_res18_v1_fit_bwd(args):
    args = la_res18_v1_fit(args)

    args.weight_decay = 1e-1
    args.expId = 'la_res18_v1_fit_bwd'
    return args


def la_res18_v1_fit_swd(args):
    args = la_res18_v1_fit(args)

    args.weight_decay = 1e-3
    args.expId = 'la_res18_v1_fit_swd'
    return args


def ir_res18_v1_load(args, loadstep=2342106):
    args.loadcolname = 'control'
    args.v4_nodes = 'encode_[2:1:11]'
    args.loadexpId = 'full'
    args.loadstep = loadstep
    args.train_num_steps = loadstep + 30000
    return args


def ir_res18_v1_fit(args):
    args = basic_res18_setting(args)
    args = basic_inst_model_setting(args)
    args = v1_cadena_fit_setting(args)
    args = ir_res18_v1_load(args)
    args.expId = 'ir_res18_v1_fit'
    return args


def ir_res18_v1_fit_bwd(args):
    args = ir_res18_v1_fit(args)

    args.weight_decay = 1e-1
    args.expId = 'ir_res18_v1_fit_bwd'
    return args


def ir_res18_v1_fit_swd(args):
    args = ir_res18_v1_fit(args)

    args.weight_decay = 1e-3
    args.expId = 'ir_res18_v1_fit_swd'
    return args
