from brainscore_mask import tf_model_loader
import json
import tensorflow as tf
import pdb


def set_var_list(args):
    ckpt_file = tf_model_loader.load_model_from_mgdb(
            db=args.load_dbname,
            col=args.load_colname,
            exp=args.load_expId,
            port=args.load_port,
            cache_dir=args.model_cache_dir,
            step_num=args.load_step,
            )
    reader = tf.train.NewCheckpointReader(ckpt_file)
    var_shapes = reader.get_variable_to_shape_map()
    var_dict = {}
    for each_var in var_shapes:
        if ('prednet' in each_var) \
                and ('layer' in each_var) \
                and ('Adam' not in each_var):
            var_dict[each_var] = each_var.strip('__GPU0__/')
    args.load_var_list = json.dumps(var_dict)
    return args


def get_prednet_layers(num_layers):
    layers = ','.join(
            ['A_%i' % i for i in range(1, num_layers)] \
            + ['Ahat_%i' % i for i in range(1, num_layers)] \
            + ['E_%i' % i for i in range(1, num_layers)] \
            + ['R_%i' % i for i in range(1, num_layers)]
            )
    return layers


def set_prednet_mgdb_info(args):
    args.load_port = 26001
    args.load_dbname = 'vd_unsup_fx'
    args.load_colname = 'prednet_fx3'
    return args


def prednet_kinetics(args):
    args.prep_type = 'no_prep'
    args.model_type = 'vd_prednet:default'
    args.batch_size = 32
    args.layers = get_prednet_layers(4)
    args = set_prednet_mgdb_info(args)
    args.load_expId = 'kinetics'
    args.load_step = 90000
    args = set_var_list(args)
    return args


def prednet_infant(args):
    args.prep_type = 'no_prep'
    args.model_type = 'vd_prednet:default'
    args.batch_size = 32
    args.layers = get_prednet_layers(4)
    args = set_prednet_mgdb_info(args)
    args.load_expId = 'infant'
    args.load_step = 90000
    args = set_var_list(args)
    return args


def prednet_infant_l9(args):
    # It's actually kinetics_l9
    args.prep_type = 'no_prep'
    args.model_type = 'vd_prednet:prednet_l9'
    args.batch_size = 32
    args.layers = get_prednet_layers(10)
    args = set_prednet_mgdb_info(args)
    args.load_expId = 'infant_l9'
    args.load_step = 100000
    args = set_var_list(args)
    return args


def filter_layers_startswith(layers, start_str):
    layers = ','.join(
            filter(
                lambda x: x.startswith(start_str), 
                layers.split(',')))
    return layers


def prednet_infant_l9_A(args):
    args = prednet_infant_l9(args)
    args.layers = filter_layers_startswith(args.layers, 'A_')
    return args


def prednet_infant_l9_Ahat(args):
    args = prednet_infant_l9(args)
    args.layers = filter_layers_startswith(args.layers, 'Ahat_')
    return args


def prednet_infant_l9_E(args):
    args = prednet_infant_l9(args)
    args.layers = filter_layers_startswith(args.layers, 'E_')
    return args


def prednet_infant_l9_R(args):
    args = prednet_infant_l9(args)
    args.layers = filter_layers_startswith(args.layers, 'R_')
    return args
