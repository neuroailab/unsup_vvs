import tensorflow as tf
from argparse import Namespace
from models.model_blocks import NoramlNetfromConv
import sys, os
import pdb
sys.path.append(os.path.expanduser('~/video_unsup/'))
import tf_model.prednet.prednet as prednet
from collections import OrderedDict


def build_all_outs(images, model_type):
    nt = 10
    images = tf.cast(images, tf.float32) / 255
    images = tf.tile(
            tf.expand_dims(images, axis=1),
            [1, nt, 1, 1, 1])
    _prednet_args = Namespace()
    _prednet_args.model_type = model_type
    prednet_inst = prednet.get_prednet_inst(_prednet_args, unroll=True)
    _ = prednet_inst(images)
    raw_all_outputs = prednet_inst.all_outputs
    all_outs = {}
    nb_layers = prednet_inst.nb_layers
    for module_name in raw_all_outputs[0]:
        for layer_idx in range(nb_layers):
            _curr_module_layer = [\
                    raw_all_outputs[t_idx][module_name][layer_idx] \
                    for t_idx in range(nt)]
            all_outs[f'{module_name}_{layer_idx}'] = sum(_curr_module_layer) / nt
    return all_outs


def build(inputs, train=True, model_type='default', which_layer='A_9', **kwargs):
    all_outs = build_all_outs(inputs['image_imagenet'], model_type)
    output = all_outs[which_layer]
    output = tf.reshape(output, [output.shape[0], -1])
    m = NoramlNetfromConv(seed=0)
    with tf.variable_scope('category_trans'):
        cate_out = m.fc(
                out_shape=1000,
                init='xavier',
                weight_decay=1e-4,
                activation=None,
                bias=0,
                dropout=None,
                in_layer=output,
                )
    ret_outputs = OrderedDict()
    ret_outputs['imagenet'] = OrderedDict()
    ret_outputs['imagenet']['category'] = cate_out
    return ret_outputs, {}


def l3_build(
        inputs, train=True, 
        wanted_layers = ['A_3', 'Ahat_3', 'E_3', 'R_3'],
        pool_ksize=4, pool_stride=4,
        tpu=False,
        **kwargs):
    all_outs = build_all_outs(inputs['image_imagenet'], 'default')
    m = NoramlNetfromConv(seed=0)
    cate_out = []
    with tf.variable_scope('category_trans'):
        for each_layer in wanted_layers:
            with tf.variable_scope(each_layer):
                curr_out = all_outs[each_layer]
                curr_out = m.pool(
                        ksize=pool_ksize,
                        stride=pool_stride,
                        pfunc='avgpool',
                        in_layer=curr_out,
                        )
                curr_out = tf.reshape(
                        curr_out, [curr_out.shape[0], -1])
                curr_out = m.fc(
                        out_shape=1000,
                        init='xavier',
                        weight_decay=1e-4,
                        activation=None,
                        bias=0,
                        dropout=None,
                        in_layer=curr_out,
                        )
                cate_out.append(curr_out)
    ret_outputs = OrderedDict()
    ret_outputs['imagenet'] = OrderedDict()
    ret_outputs['imagenet']['category'] = cate_out
    if not tpu:
        return ret_outputs, {}
    else:
        return tf.stack(cate_out, axis=1)
