import os
import sys
import pdb
import tensorflow as tf
from collections import OrderedDict

import models.simclr.resnet as resnet
from models.model_blocks import NoramlNetfromConv
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'global_bn', True,
    'Whether to aggregate BN statistics across distributed cores.')

flags.DEFINE_float(
    'batch_norm_decay', 0.9,
    'Batch norm decay parameter.')

flags.DEFINE_integer(
    'fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linera head.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')


def build(inputs, train=True, **kwargs):
    FLAGS(['none'])
    with tf.variable_scope('base_model'):
        model = resnet.resnet_v1(
                resnet_depth=18,
                width_multiplier=1,
                cifar_stem=False)
        output = model(
                tf.cast(inputs['image_imagenet'], tf.float32) / 255, 
                is_training=False)
    ending_points = resnet.ENDING_POINTS
    output = ending_points[-1]
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


def build_with_mid(inputs, train=True, **kwargs):
    FLAGS(['none'])
    with tf.variable_scope('base_model'):
        model = resnet.resnet_v1(
                resnet_depth=18,
                width_multiplier=1,
                cifar_stem=False)
        output = model(
                tf.cast(inputs['image_imagenet'], tf.float32) / 255, 
                is_training=False)
    ending_points = resnet.ENDING_POINTS
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
