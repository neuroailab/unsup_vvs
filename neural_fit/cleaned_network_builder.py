import os
import sys
import pdb
import tensorflow as tf
from argparse import Namespace
from collections import OrderedDict

from tfutils.imagenet_data import color_normalize
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../network_training')) # needed for inside imports
sys.path.append(os.path.abspath('../network_training/models')) # needed for inside imports
from models.model_blocks import NoramlNetfromConv
import network_training.cmd_parser as cmd_parser
from network_training.models.config_parser import get_network_cfg
from network_training.models.model_builder import ModelBuilder
from network_training.models.rp_col_utils import rgb_to_lab
from network_training.models.mean_teacher_utils import \
        ema_variable_scope, name_variable_scope
from network_training.models.instance_task.model.instance_model \
        import resnet_embedding

MEAN_RGB = [0.485, 0.456, 0.406]


def get_simclr_ending_points(inputs):
    sys.path.append(
            os.path.abspath('../network_training/models/simclr'))
    import resnet, run
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(['none'])
    with tf.variable_scope('base_model'):
        model = resnet.resnet_v1(
                resnet_depth=18,
                width_multiplier=1,
                cifar_stem=False,
                drop_final_pool=True)
        output = model(
                tf.cast(inputs['images'], tf.float32) / 255, 
                is_training=False)
    ending_points = resnet.ENDING_POINTS
    return ending_points


def get_network_cfg_from_setting(setting_name=None, **kwargs):
    train_args = Namespace()

    # Required fields for ModelBuilder
    train_args.seed = None
    train_args.weight_decay = 0.0
    train_args.global_weight_decay = 0.0
    train_args.enable_bn_weight_decay = False
    train_args.ignorebname_new = 1
    train_args.add_batchname = None
    train_args.tpu_flag = 1
    train_args.combine_tpu_flag = 0
    train_args.tpu_task = None
    train_args.init_stddev = .01
    train_args.init_type = 'xavier'
    train_args.sm_bn_trainable = 0
    train_args.sm_resnetv2 = 0
    train_args.sm_resnetv2_1 = 0

    for key, value in kwargs.items():
        setattr(train_args, key, value)

    if setting_name is not None:
        train_args.load_setting_func = setting_name
        train_args = cmd_parser.load_setting(train_args)
        if not hasattr(train_args, 'network_func_kwargs'):
            train_args.network_func_kwargs = None
    else:
        assert (hasattr(train_args, 'network_func') \
                or hasattr(train_args, 'pathconfig')), \
                'Must provide network cfg'
        if not hasattr(train_args, 'network_func'):
            train_args.network_func = None

    network_cfg = get_network_cfg(train_args)
    return network_cfg, train_args


def build_vm_model_from_args(args, post_input_image, module_name=['encode']):
    model_builder = ModelBuilder(args, {})

    # Preparations before building one module
    model_builder.dataset_prefix = 'imagenet'
    model_builder.reuse_dict = {}
    model_builder.init_model_block_class()
    model_builder.train = False
    model_builder.all_out_dict = {}
    model_builder.outputs_dataset = OrderedDict()
    model_builder.save_layer_middle_output = True

    for each_module_name in module_name:
        model_builder.build_one_module(each_module_name, post_input_image)
    return model_builder.all_out_dict


def get_network_outputs(
        inputs, 
        prep_type,
        model_type,
        setting_name=None,
        module_name=['encode'],
        inst_resnet_size=18,
        **cfg_kwargs):
    input_image = inputs['images']
    if prep_type == 'only_mean':
        input_image = tf.cast(input_image, tf.float32) / 255
        offset = tf.constant(MEAN_RGB, shape=[1, 1, 1, 3])
        post_input_image = input_image - offset
    elif prep_type == 'mean_std':
        # Divided by 255 is done inside the function
        post_input_image = color_normalize(input_image)
    elif prep_type == 'color_prep':
        input_image = tf.cast(input_image, tf.float32) / 255
        post_input_image = rgb_to_lab(input_image)
        post_input_image = post_input_image[:,:,:,:1] - 50
    elif prep_type == 'no_prep':
        post_input_image = tf.cast(input_image, tf.float32) / 255
    else:
        raise NotImplementedError('Preprocessing type not supported!')

    if model_type == 'vm_model':
        network_cfg, args = get_network_cfg_from_setting(setting_name, **cfg_kwargs)
        all_outs = build_vm_model_from_args(args, post_input_image, module_name)
    elif model_type == 'mt_vm_model':
        network_cfg, args = get_network_cfg_from_setting(setting_name, **cfg_kwargs)
        with name_variable_scope("primary", "primary", reuse=tf.AUTO_REUSE) \
                as (name_scope, var_scope):
            all_outs = build_vm_model_from_args(
                    args, post_input_image, module_name)
        with ema_variable_scope("ema", var_scope, reuse=tf.AUTO_REUSE):
            ema_out_dict = build_vm_model_from_args(
                    args, post_input_image, module_name)
            # Combine two out dicts, adding ema_ as prefix to keys
            for each_key, each_value in ema_out_dict.items():
                new_key = "ema_%s" % each_key
                assert not new_key in all_outs, \
                        "New key %s already exists" % new_key
                all_outs[new_key] = each_value
    elif model_type.startswith('inst_model'):
        # inst_model:get_all_layers_args
        if ':' not in model_type:
            get_all_layers = 'all_spatial'
        else:
            get_all_layers = model_type.split(':')[1]
        all_outs = resnet_embedding(
                inputs['images'],
                get_all_layers=get_all_layers,
                skip_final_dense=True,
                resnet_size=inst_resnet_size)
    elif model_type.startswith('vd_inst_model'):
        # model_type should be vd_inst_model:actual_model_type
        sys.path.append(os.path.expanduser('~/video_unsup/'))
        import tf_model.model.instance_model as vd_inst_model
        _model_type = model_type.split(':')[1]
        input_images = inputs['images']
        if _model_type == '3dresnet':
            curr_shape = input_images.get_shape().as_list()[1]
            input_images = tf.image.resize_images(
                    input_images, [curr_shape // 2, curr_shape // 2])
            input_images = tf.tile(
                    tf.expand_dims(input_images, axis=1),
                    [1, 16, 1, 1, 1])
        elif _model_type == '3dresnet_full':
            input_images = tf.tile(
                    tf.expand_dims(input_images, axis=1),
                    [1, 16, 1, 1, 1])
            _model_type = '3dresnet'
        elif _model_type == 'vanilla3d_single':
            input_images = tf.tile(
                    tf.expand_dims(input_images, axis=1),
                    [1, 16, 1, 1, 1])
        all_outs = vd_inst_model.resnet_embedding(
                input_images,
                get_all_layers='all_spatial',
                skip_final_dense=True,
                model_type=_model_type)
    elif model_type.startswith('vd_prednet'):
        # model_type should be vd_prednet:actual_model_type
        _model_type = model_type.split(':')[1]
        images = inputs['images']
        if _model_type == 'prednet_l9':
            curr_shape = images.get_shape().as_list()[1]
            if not curr_shape % 32 == 0:
                new_shape = curr_shape // 32 * 32
                images = tf.image.resize_images(
                        images, [new_shape, new_shape])
        import network_training.models.prednet_builder as prednet_builder
        all_outs = prednet_builder.build_all_outs(images, _model_type)
    elif model_type == 'simclr_model':
        ending_points = get_simclr_ending_points(inputs)
        all_outs = {
                'encode_%i' % (_idx+1): _rep \
                for _idx, _rep in enumerate(ending_points)
                }
    elif model_type == 'simclr_model_mid':
        ending_points = get_simclr_ending_points(inputs)
        output = ending_points[-1]
        output = tf.reshape(output, [output.shape[0], -1])
        m = NoramlNetfromConv(seed=0)
        with tf.variable_scope('category_trans'):
            with tf.variable_scope('mid'):
                output = m.fc(
                        out_shape=1000,
                        init='xavier',
                        weight_decay=1e-4,
                        activation='relu',
                        bias=0.1,
                        dropout=None,
                        in_layer=output,
                        )
        all_outs = {'category_trans_1': output}
    else:
        raise NotImplementedError('Model type not supported!')
    all_outs['model_inputs'] = post_input_image
    return all_outs, {}


def test_vm_model():
    setting_name = 'part10_cate_res18_fix'
    input_image = tf.zeros([1, 224, 224, 3], dtype=tf.float32)
    network_outputs = get_network_outputs(
            {'images': input_image}, 
            prep_type='mean_std',
            model_type='vm_model',
            setting_name=setting_name)
    print(network_outputs)


def test_inst_model():
    input_image = tf.zeros([1, 224, 224, 3], dtype=tf.float32)
    network_outputs = get_network_outputs(
            {'images': input_image}, 
            prep_type='no_prep',
            model_type='inst_model',
            )
    print(network_outputs)


def test_video_model():
    input_image = tf.zeros([1, 224, 224, 3], dtype=tf.float32)
    network_outputs = get_network_outputs(
            {'images': input_image}, 
            prep_type='no_prep',
            model_type='vd_inst_model:3dresnet',
            )
    print(network_outputs)


def test_simclr_model():
    input_image = tf.zeros([1, 224, 224, 3], dtype=tf.float32)
    network_outputs = get_network_outputs(
            {'images': input_image}, 
            prep_type='no_prep',
            model_type='simclr_model',
            )
    print(network_outputs)


if __name__ == '__main__':
    #test_vm_model()
    #test_inst_model()
    #test_video_model()
    test_simclr_model()
