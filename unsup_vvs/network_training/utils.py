from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np
import tensorflow as tf
import copy
import pdb
import json

from utilities.data_path_utils import which_imagenet_map, \
        get_data_path, \
        get_TPU_data_path
from models.rp_col_utils import rgb_to_lab, ab_to_Q, \
        pos2lbl, lab_to_rgb, Q_to_ab
from models.tpu_loss_utils import metric_fn, rp_metric_fn, depth_metric_fn, \
        combine_depth_imn_metric_fn, col_metric_fn, tpu_instance_metric_fn, \
        combine_rp_imn_metric_fn, tpu_imagenet_loss, tpu_rp_imagenet_loss, \
        tpu_col_loss, tpu_depth_loss, combine_depth_imn_loss, \
        combine_rp_imn_loss, tpu_mean_teacher_metric_fn, \
        tpu_mean_teacher_loss
from models.loss_utils import instance_loss, get_cons_coefficient, \
        sigmoid_rampup, mean_teacher_consitence_and_res
from models.mean_teacher_utils import rampup_rampdown_lr, \
        name_variable_scope, ema_variable_scope
from models.config_parser import get_network_cfg, postprocess_config


def get_val_target(cfg_dataset):
    val_target = []
    if cfg_dataset.get('scenenet', 0)==1:
        need_normal = cfg_dataset.get('scene_normal', 1)==1
        need_depth = cfg_dataset.get('scene_depth', 1)==1
        need_instance = cfg_dataset.get('scene_instance', 0)==1
        #val_target.extend(['normal_scenenet', 'depth_scenenet'])
        if need_normal:
            val_target.append('normal_scenenet')
        if need_depth:
            val_target.append('depth_scenenet')
        if need_instance:
            val_target.append('instance_scenenet')

    if cfg_dataset.get('scannet', 0)==1:
        val_target.extend(['depth_scannet'])

    if cfg_dataset.get('pbrnet', 0)==1:
        need_normal = cfg_dataset.get('pbr_normal', 1)==1
        need_depth = cfg_dataset.get('pbr_depth', 1)==1
        need_instance = cfg_dataset.get('pbr_instance', 0)==1

        #val_target.extend(['normal_pbrnet', 'depth_pbrnet'])
        if need_normal:
            val_target.append('normal_pbrnet')
        if need_depth:
            val_target.append('depth_pbrnet')
        if need_instance:
            val_target.append('instance_pbrnet')

    if cfg_dataset.get('imagenet', 0)==1:
        val_target.extend(['label_imagenet'])

    if cfg_dataset.get('coco', 0)==1:
        val_target.append('mask_coco')

    if cfg_dataset.get('place', 0)==1:
        val_target.extend(['label_place'])

    if cfg_dataset.get('kinetics', 0)==1:
        val_target.extend(['label_kinetics'])

    if cfg_dataset.get('nyuv2', 0)==1:
        val_target.extend(['depth_nyuv2'])

    return val_target


def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res


def mean_losses_keep_rest(step_results):
    retval = {}
    keys = step_results[0].keys()
    for k in keys:
        plucked = [d[k] for d in step_results]
        if isinstance(k, str) and 'loss' in k:
            retval[k] = np.mean(plucked)
        else:
            retval[k] = plucked
    return retval


def preprocess_config(cfg):
    cfg = copy.deepcopy(cfg)
    for k in cfg.get("network_list", ["decode", "encode"]):
        if k in cfg:
            ks = cfg[k].keys()
            for _k in ks:
                #assert isinstance(_k, int), _k
                cfg[k][str(_k)] = cfg[k].pop(_k)
    return cfg

'''
This is Siming's working park!
'''

def gpu_col_loss(outputs, *args, **kwargs):
    #print("col loss input:", outputs)
    soft=True
    if soft:
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs['logits'], labels=outputs['Q'])
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs['logits'], labels=outputs['Q'])

    return loss

def gpu_col_tl_loss(outputs, *args, **kwargs):
    print("##########col loss input:###########", outputs)
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs['logits'], labels=outputs['Q'])
    one_hot_labels = tf.one_hot(outputs['Q'], 1000)
    imnet_loss = tf.losses.softmax_cross_entropy(logits=outputs['logits'], onehot_labels=one_hot_labels)

    top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(tf.reshape(outputs['logits'], [-1, outputs['logits'].get_shape().as_list()[-1]]),tf.reshape(outputs['Q'], [-1]), 1), tf.float32))
    top1 = tf.Print(top1, [top1], message="Top1")
    imnet_loss = imnet_loss + top1
    imnet_loss = imnet_loss - top1
    return imnet_loss

def gpu_col_val(inputs, outputs, *args):
    #print("gpu_col_val inputs:", inputs)
    #print("gpu_col_val outputs:", outputs)
    soft=True
    if soft:
        outputs['Q'] = tf.argmax(outputs['Q'], -1)

    return {'top1': tf.reduce_mean(tf.cast(tf.nn.in_top_k(tf.reshape(outputs['logits'], [-1, outputs['logits'].get_shape().as_list()[-1]]),tf.reshape(outputs['Q'], [-1]), 1), tf.float32))
            }

def gpu_col_tl_val(inputs, outputs, *args):
    print("gpu_col_val inputs:", inputs)
    print("gpu_col_val outputs:", outputs)
    return {'top1': tf.reduce_mean(tf.cast(tf.nn.in_top_k(tf.reshape(outputs['logits'], [-1, outputs['logits'].get_shape().as_list()[-1]]),tf.reshape(outputs['Q'], [-1]), 1), tf.float32))
            }
