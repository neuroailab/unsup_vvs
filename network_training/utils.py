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

# TODO: use this instead of l2 loss
def loss_ave_invdot(output, label_0, label_1):
    def _process(label):
        label = tf.cast(label, tf.float32)
        label = tf.div(label, tf.constant(255, dtype=tf.float32))
        return label
    output_0 = tf.nn.l2_normalize(output[0], 3)
    labels_0 = tf.nn.l2_normalize(_process(label_0), 3)
    loss_0 = -tf.reduce_sum(tf.multiply(output_0, labels_0)) \
            / np.prod(label_0.get_shape().as_list()) * 3

    output_1 = tf.nn.l2_normalize(output[1], 3)
    labels_1 = tf.nn.l2_normalize(_process(label_1), 3)
    loss_1 = -tf.reduce_sum(tf.multiply(output_1, labels_1)) \
            / np.prod(label_1.get_shape().as_list()) * 3

    loss = tf.add(loss_0, loss_1)
    return loss


def normal_loss(output, label, normalloss = 0, sm_half_size = 0):
    def _process(label):
        if sm_half_size == 1:
            print("****************resize_label****************")
            print(label.shape)
            label = tf.image.resize_images(label, (112, 112))
        label = tf.cast(label, tf.float32)
        label = tf.div(label, tf.constant(255, dtype=tf.float32))
        return label

    curr_label = _process(label)
    if normalloss==0:
        curr_loss = tf.nn.l2_loss(output - curr_label) / np.prod(curr_label.get_shape().as_list())
    elif normalloss==1:
        curr_label = tf.nn.l2_normalize(curr_label, 3)
        curr_output = tf.nn.l2_normalize(output, 3)
        curr_loss = -tf.reduce_sum(tf.multiply(curr_label, curr_output)) / np.prod(curr_label.get_shape().as_list()) * 3
    else:
        curr_label = tf.nn.l2_normalize(curr_label*255/128 - 1, 3)
        curr_output = tf.nn.l2_normalize(output, 3)
        curr_loss = -tf.reduce_sum(tf.multiply(curr_label, curr_output)) / np.prod(curr_label.get_shape().as_list()) * 3

    return curr_loss

def l2_loss_withmask(output, label, mask):
    mask = tf.cast(mask, tf.float32)
    #mask = tf.Print(mask, [tf.reduce_sum(mask)], message = 'Resuce sum of mask')
    return tf.nn.l2_loss(tf.multiply(output - label, mask)) / tf.reduce_sum(mask)

def dep_l2_loss_eigen(output, label):
    diff = output - label
    diff_shape = diff.get_shape().as_list()
    loss_0 = tf.nn.l2_loss(diff) / np.prod(diff_shape)
    loss_1 = tf.square(tf.reduce_sum(diff))/(2*np.prod(diff_shape)*np.prod(diff_shape))

    weight_np_x = np.zeros([3, 3, 1, 1])
    weight_np_x[1, 0, 0, 0] = 0.5
    weight_np_x[1, 2, 0, 0] = -0.5

    weight_conv2d_x = tf.constant(weight_np_x, dtype = tf.float32)

    weight_np_y = np.zeros([3, 3, 1, 1])
    weight_np_y[0, 1, 0, 0] = 0.5
    weight_np_y[2, 1, 0, 0] = -0.5

    weight_conv2d_y = tf.constant(weight_np_y, dtype = tf.float32)

    tmp_dx = tf.nn.conv2d(diff, weight_conv2d_x,
                strides=[1, 1, 1, 1],
                padding='SAME')
    tmp_dy = tf.nn.conv2d(diff, weight_conv2d_y,
                strides=[1, 1, 1, 1],
                padding='SAME')

    loss_2 = tf.reduce_sum(tf.add(tf.square(tmp_dx), tf.square(tmp_dy)))/np.prod(tmp_dx.get_shape().as_list())
    final_loss = loss_0 - loss_1 + loss_2

    return final_loss

def dep_loss_berHu(output, label):
    diff = output - label
    diff_shape = diff.get_shape().as_list()
    diff = tf.abs(diff)

    diff_c = 1.0/5*tf.reduce_max(diff)
    curr_mask = tf.less_equal(diff, diff_c)
    tmp_mask_0 = tf.cast(curr_mask, tf.float32)
    tmp_mask_1 = tf.cast(tf.logical_not(curr_mask), tf.float32)

    tmp_l2_loss = (tf.square(tf.multiply(diff, tmp_mask_1)) + tf.square(diff_c))/(2*diff_c)
    tmp_l1_loss = tf.multiply(diff, tmp_mask_0)
    loss = ( tf.reduce_sum(tmp_l1_loss) + tf.reduce_sum(tmp_l2_loss)) / np.prod(diff_shape)

    return loss

def depth_loss(output, label, depthloss = 0, depth_norm = 8000, sm_half_size = 0):

    def _process_dep(label):
        print("*******depth label shape*********")
        print(label.shape)
        if sm_half_size == 1:
            print("*****half size******")
            label = tf.image.resize_images(label, (112, 112))
        label = tf.cast(label, tf.float32)
        label = tf.div(label, tf.constant(depth_norm, dtype=tf.float32))
        return label

    curr_label = _process_dep(label)
    if depthloss==0:
        curr_loss = tf.nn.l2_loss(output - curr_label) / np.prod(curr_label.get_shape().as_list())
    elif depthloss==1: # loss from Eigen, Fergus 2015
        curr_loss = dep_l2_loss_eigen(output, curr_label)
    elif depthloss==2:
        curr_loss = l2_loss_withmask(output, curr_label, tf.not_equal(curr_label, tf.constant(0, tf.float32)))
    elif depthloss==3:
        curr_loss = dep_loss_berHu(output, curr_label)

    return curr_loss

def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res

def get_semantic_loss(
        curr_predict, curr_truth, 
        need_mask=True, mask_range=40, less_or_large=0):
    curr_shape = curr_predict.get_shape().as_list()
    curr_predict = tf.reshape(curr_predict, [-1, curr_shape[-1]])

    curr_truth = tf.reshape(curr_truth, [-1])
    curr_truth = tf.cast(curr_truth, tf.int32)

    if need_mask:
        if less_or_large==0:
            truth_mask = tf.less(curr_truth, mask_range)
        else:
            truth_mask = tf.greater(curr_truth, mask_range)

        curr_truth = tf.boolean_mask(curr_truth, truth_mask)
        curr_predict = tf.boolean_mask(curr_predict, truth_mask)

    curr_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=curr_predict, 
            labels=curr_truth)

    return curr_loss


def get_softmax_loss(curr_label, curr_output, label_norm, multtime):
    if multtime==1:
        curr_label = tf.reshape(curr_label, [-1])
        curr_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = curr_output, labels = curr_label)
        curr_loss = tf.div(
                curr_loss, 
                tf.constant(label_norm, dtype = tf.float32))
        curr_loss = tf.reduce_mean(curr_loss)
    else:
        list_output = tf.split(
                curr_output, axis = 1, 
                num_or_size_splits = multtime)
        curr_list_loss = [
                get_softmax_loss(
                    curr_label, 
                    _curr_output, 
                    label_norm=label_norm, 
                    multtime=1) 
                for _curr_output in list_output]
        curr_loss = tf.reduce_mean(curr_list_loss)

    return curr_loss


def loss_withcfg(output, *args, **kwargs):
    cfg_dataset = kwargs.get('cfg_dataset', {})
    depth_norm = kwargs.get('depth_norm', 8000)
    label_norm = kwargs.get('label_norm', 20)
    depthloss = kwargs.get('depthloss', 0)
    normalloss = kwargs.get('normalloss', 0)
    multtime = kwargs.get('multtime', 1)
    extra_feat = kwargs.get('extra_feat', 0)
    sm_half_size = kwargs.get('sm_half_size', 0)
    mean_teacher = kwargs.get('mean_teacher', False)
    res_coef = kwargs.get('res_coef', 0.01)
    cons_ramp_len = kwargs.get('cons_ramp_len', 400000)
    cons_max_value = kwargs.get('cons_max_value', 10.0)
    instance_task = kwargs.get('instance_task', False)
    instance_k = kwargs.get('instance_k', 4096)
    instance_data_len = kwargs.get('instance_data_len', 1281025)
    inst_and_cate = kwargs.get('inst_and_cate', False)

    assert not (inst_and_cate and mean_teacher), "Not supported yet!"

    cons_coefficient = get_cons_coefficient(cons_ramp_len, cons_max_value)
    
    now_indx = 0
    loss_list = []
    arg_offset = 0
    if cfg_dataset.get('scenenet', 0)==1:
        if cfg_dataset.get('scene_normal', 1)==1:
            curr_loss = normal_loss(
                    output[now_indx], 
                    args[now_indx + arg_offset], 
                    normalloss=normalloss, 
                    sm_half_size=sm_half_size)

            loss_list.append(curr_loss)
            now_indx = now_indx + 1

        if cfg_dataset.get('scene_depth', 1)==1:
            curr_loss = depth_loss(
                    output[now_indx], 
                    args[now_indx + arg_offset], 
                    depthloss=depthloss, 
                    depth_norm=depth_norm, 
                    sm_half_size=sm_half_size)

            loss_list.append(curr_loss)
            now_indx = now_indx + 1

        if cfg_dataset.get('scene_instance', 0)==1:
            curr_loss = get_semantic_loss(
                    curr_predict = output[now_indx], 
                    curr_truth = args[now_indx + arg_offset], 
                    need_mask = False)
            curr_loss = tf.reduce_mean(curr_loss)
            loss_list.append(curr_loss)
            now_indx = now_indx + 1

    if cfg_dataset.get('pbrnet', 0)==1:
        if cfg_dataset.get('pbr_normal', 1)==1:
            curr_loss = normal_loss(
                    output[now_indx], args[now_indx + arg_offset], 
                    normalloss = normalloss, sm_half_size=sm_half_size)

            loss_list.append(curr_loss)
            now_indx = now_indx + 1

        if cfg_dataset.get('pbr_depth', 1)==1:
            curr_loss = depth_loss(
                    output[now_indx], args[now_indx + arg_offset], 
                    depthloss = depthloss, depth_norm = depth_norm, 
                    sm_half_size=sm_half_size)

            loss_list.append(curr_loss)
            now_indx = now_indx + 1

        if cfg_dataset.get('pbr_instance', 0)==1:
            curr_loss = get_semantic_loss(
                    curr_predict = output[now_indx], 
                    curr_truth = args[now_indx + arg_offset], 
                    need_mask = True, mask_range = 40)
            curr_loss = tf.reduce_mean(curr_loss)
            loss_list.append(curr_loss)
            now_indx = now_indx + 1

    num_inst_outputs = 6
    if cfg_dataset.get('imagenet', 0)==1 \
            or cfg_dataset.get('rp', 0)==1 \
            or cfg_dataset.get('colorization', 0)==1:
        curr_loss = 0
        if not instance_task or inst_and_cate or mean_teacher:
            # Default behavior
            curr_loss = get_softmax_loss(
                    curr_label=args[now_indx + arg_offset], 
                    curr_output=output[now_indx], 
                    label_norm=label_norm, multtime=multtime)

        if instance_task and not mean_teacher:
            # Compute the NCE loss for instance task
            output_offset = int(inst_and_cate)
            data_dist = output[now_indx + output_offset]
            noise_dist = output[now_indx + 1 + output_offset]
            inst_loss, _, _ = instance_loss(
                    data_dist, noise_dist, 
                    instance_k, instance_data_len
                    )
            arg_offset -= num_inst_outputs
            now_indx += num_inst_outputs
            curr_loss += inst_loss

        if mean_teacher:
            if instance_task:
                ema_output_offset = num_inst_outputs
            else:
                ema_output_offset = 0

            consistence_loss, res_loss = mean_teacher_consitence_and_res(
                    class_logit=output[now_indx],
                    cons_logit=output[now_indx +1],
                    ema_class_logit=output[now_indx + 2 + ema_output_offset],
                    cons_coefficient=cons_coefficient,
                    res_coef=res_coef,
                    )
            
            curr_loss += consistence_loss + res_loss # Add all three losses
            # additional 3 more outputs, labels not moved
            arg_offset -= 3 + ema_output_offset*2
            now_indx += 3 + ema_output_offset*2

        loss_list.append(curr_loss)
        # Update the index for args
        now_indx = now_indx + 1
        if extra_feat==1:
            # As we don't need normal and depth here, skip
            now_indx = now_indx + 2
            arg_offset = arg_offset - 2

    if cfg_dataset.get('imagenet_un', 0)==1 and mean_teacher:
        ema_output_offset = 0
        curr_loss = 0
        if instance_task:
            ema_output_offset = num_inst_outputs
            # Compute the NCE loss for instance task
            output_offset = 2
            data_dist = output[now_indx + output_offset]
            noise_dist = output[now_indx + 1 + output_offset]
            inst_loss, _, _ = instance_loss(
                    data_dist, noise_dist, 
                    instance_k, instance_data_len
                    )
            curr_loss += inst_loss

        consistence_loss, res_loss = mean_teacher_consitence_and_res(
                class_logit=output[now_indx],
                cons_logit=output[now_indx + 1],
                ema_class_logit=output[now_indx + 2 + ema_output_offset],
                cons_coefficient=cons_coefficient,
                res_coef=res_coef,
                )
        curr_loss += consistence_loss + res_loss # Add all three losses
        arg_offset -= 4 + ema_output_offset
        now_indx += 4 + ema_output_offset

        loss_list.append(curr_loss)

    if cfg_dataset.get('coco', 0)==1:
        curr_loss = get_semantic_loss(
                curr_predict = output[now_indx], 
                curr_truth = args[now_indx + arg_offset], 
                need_mask = True, mask_range = 0, less_or_large = 1)
        curr_loss = tf.reduce_mean(curr_loss)

        loss_list.append(curr_loss)
        now_indx = now_indx + 1

    if cfg_dataset.get('place', 0)==1:
        curr_loss = get_softmax_loss(
                curr_label = args[now_indx + arg_offset], 
                curr_output = output[now_indx], 
                label_norm = label_norm, multtime = multtime)

        loss_list.append(curr_loss)
        now_indx = now_indx + 1

        if extra_feat==1:
            # As we don't need normal and depth here, skip
            now_indx = now_indx + 2
            arg_offset = arg_offset - 2

    if cfg_dataset.get('kinetics', 0)==1:
        curr_loss = get_softmax_loss(
                curr_label = args[now_indx + arg_offset], 
                curr_output = output[now_indx], 
                label_norm = label_norm, multtime = multtime)

        loss_list.append(curr_loss)
        now_indx = now_indx + 1

    if cfg_dataset.get('nyuv2', 0)==1:
        curr_loss = depth_loss(
                output[now_indx], args[now_indx + arg_offset], 
                depthloss = depthloss, depth_norm = depth_norm)

        loss_list.append(curr_loss)
        now_indx = now_indx + 1

    return tf.add_n(loss_list)


def add_topn_report(
        curr_label, curr_output, label_norm, 
        top_or_loss, multtime, loss_dict, str_suffix = 'imagenet'):

    if multtime==1:
        curr_label = tf.reshape(curr_label, [-1])
        if top_or_loss==0:
            curr_top1 = tf.nn.in_top_k(curr_output, curr_label, 1)
            curr_top5 = tf.nn.in_top_k(curr_output, curr_label, 5)
            loss_dict['loss_top1_%s' % str_suffix] = curr_top1
            loss_dict['loss_top5_%s' % str_suffix] = curr_top5
        else:
            curr_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits = curr_output, labels = curr_label)
            curr_loss = tf.div(
                    curr_loss, 
                    tf.constant(label_norm, dtype = tf.float32))
            loss_dict['loss_%s' % str_suffix] = curr_loss
    else:
        list_output = tf.split(
                curr_output, axis = 1, num_or_size_splits = multtime)
        for _curr_indx, _curr_output in enumerate(list_output):
            loss_dict = add_topn_report(
                    curr_label = curr_label, curr_output = _curr_output, 
                    label_norm = label_norm, 
                    top_or_loss = top_or_loss, multtime = 1, 
                    loss_dict = loss_dict, 
                    str_suffix = '%i_%s' % (_curr_indx, str_suffix))

    return loss_dict


def inst_loss_with_updates(
        data_dist,
        noise_dist,
        memory_bank_list,
        all_label_list,
        data_indx,
        new_memory,
        data_label,
        devices,
        instance_k,
        instance_data_len,
        ):
    if not isinstance(memory_bank_list, tuple):
        memory_bank_list = [memory_bank_list]
    if not isinstance(all_label_list, tuple):
        all_label_list = [all_label_list]

    all_update_ops = []
    for gpu_device, all_label, memory_bank \
            in zip(devices, all_label_list, memory_bank_list):
        with tf.device(gpu_device):
            ## Update label and memory vector here
            lb_update_op = tf.scatter_update(
                    all_label, data_indx, 
                    tf.cast(data_label, tf.int64))
            mb_update_op = tf.scatter_update(
                    memory_bank, data_indx, 
                    new_memory)

            all_update_ops.append(lb_update_op)
            all_update_ops.append(mb_update_op)

    with tf.control_dependencies(all_update_ops):
        _, loss_model, loss_noise = instance_loss(
                data_dist, noise_dist, 
                instance_k, instance_data_len
                )
    return loss_model, loss_noise


def rep_losses(
        inputs,
        output,
        instance_k,
        instance_data_len,
        devices,
        inst_and_cate=False,
        inst_cate_sep=False,
        label_norm=1,
        multtime=1,
        mean_teacher=False,
        res_coef=0.01,
        cons_ramp_len=400000,
        cons_max_value=10.0,
        **kwargs
        ):
    assert not (inst_and_cate and mean_teacher), "Not supported yet!"
    cons_coefficient = get_cons_coefficient(cons_ramp_len, cons_max_value)
    ret_dict = {}

    # Record the category loss
    if inst_and_cate or mean_teacher:
        cate_loss = get_softmax_loss(
                curr_label=inputs['label_imagenet'], 
                curr_output=output[0], 
                label_norm=label_norm, multtime=multtime)
        ret_dict['loss_cate'] = cate_loss

    # Compute the NCE loss for instance task and update the memory banks
    output_offset = 0
    if inst_and_cate:
        output_offset = 1
    if not inst_cate_sep:
        data_label = inputs['label_imagenet']
    else:
        data_label = inputs['label_imagenet_un']
    if mean_teacher:
        output_offset = 18 # 2 + 6 + 2 + 6 + 2
        data_label = inputs['label_imagenet_un']

    loss_model, loss_noise = inst_loss_with_updates(
            data_dist=output[0 + output_offset],
            noise_dist=output[1 + output_offset],
            memory_bank_list=output[2 + output_offset],
            all_label_list=output[5 + output_offset],
            data_indx=output[3 + output_offset],
            new_memory=output[4 + output_offset],
            data_label=data_label,
            devices=devices,
            instance_k=instance_k,
            instance_data_len=instance_data_len,
            )
    ret_dict.update({
            'loss_model': loss_model,
            'loss_noise': loss_noise,
            })

    # Get mean teacher loss
    if mean_teacher:
        consistence_loss_0, res_loss_0 = mean_teacher_consitence_and_res(
                class_logit=output[0],
                cons_logit=output[1],
                ema_class_logit=output[8],
                cons_coefficient=cons_coefficient,
                res_coef=res_coef,
                )
        consistence_loss_1, res_loss_1 = mean_teacher_consitence_and_res(
                class_logit=output[16],
                cons_logit=output[17],
                ema_class_logit=output[24],
                cons_coefficient=cons_coefficient,
                res_coef=res_coef,
                )
        ret_dict.update({
                'mt_con': (consistence_loss_0 + consistence_loss_1)/2,
                'mt_res': (res_loss_0 + res_loss_1)/2,
                })
    return ret_dict


def rep_loss_withcfg(
        inputs, 
        output, 
        target, 
        cfg_dataset={}, 
        depth_norm=8000, 
        depthloss=0, 
        normalloss=0,
        label_norm=20, 
        top_or_loss=0, 
        multtime=1,
        extra_feat=0,
        sm_half_size=0,
        mean_teacher=False,
        instance_task=False,
        inst_and_cate=False,
        inst_cate_sep=False,
    ):
    assert not (inst_and_cate and mean_teacher), "Not supported yet!"
    now_indx = 0
    loss_dict = {}
    arg_offset = 0

    if cfg_dataset.get('scenenet', 0)==1:
        if cfg_dataset.get('scene_normal', 1)==1:
            curr_loss = normal_loss(
                    output[now_indx], 
                    inputs[target[now_indx + arg_offset]], 
                    normalloss=normalloss, 
                    sm_half_size=sm_half_size)

            loss_dict['loss_normal_scenenet'] = curr_loss
            now_indx = now_indx + 1

        if cfg_dataset.get('scene_depth', 1)==1:
            curr_loss = depth_loss(
                    output[now_indx], 
                    inputs[target[now_indx + arg_offset]], 
                    depthloss=depthloss, 
                    depth_norm=depth_norm, 
                    sm_half_size=sm_half_size)
            loss_dict['loss_depth_scenenet'] = curr_loss
            now_indx = now_indx + 1

        if cfg_dataset.get('scene_instance', 0)==1:
            curr_loss = get_semantic_loss(
                    curr_predict=output[now_indx], 
                    curr_truth=inputs[target[now_indx + arg_offset]], 
                    need_mask=False)
            loss_dict['loss_instance_scenenet'] = tf.reduce_mean(curr_loss)
            now_indx = now_indx + 1

    if cfg_dataset.get('pbrnet', 0)==1:

        if cfg_dataset.get('pbr_normal', 1)==1:
            curr_loss = normal_loss(
                    output[now_indx], 
                    inputs[target[now_indx + arg_offset]], 
                    normalloss=normalloss, 
                    sm_half_size=sm_half_size)
            loss_dict['loss_normal_pbrnet'] = curr_loss
            now_indx = now_indx + 1

        if cfg_dataset.get('pbr_depth', 1)==1:
            curr_loss = depth_loss(
                    output[now_indx], 
                    inputs[target[now_indx + arg_offset]], 
                    depthloss=depthloss, 
                    depth_norm=depth_norm, 
                    sm_half_size=sm_half_size)
            loss_dict['loss_depth_pbrnet'] = curr_loss
            now_indx = now_indx + 1

        if cfg_dataset.get('pbr_instance', 0)==1:
            curr_loss = get_semantic_loss(
                    curr_predict = output[now_indx], 
                    curr_truth=inputs[target[now_indx + arg_offset]], 
                    need_mask=True, mask_range=40)

            loss_dict['loss_instance_pbrnet'] = tf.reduce_mean(curr_loss)
            now_indx = now_indx + 1

    if cfg_dataset.get('imagenet', 0)==1:
        if not instance_task or inst_and_cate or mean_teacher:
            loss_dict = add_topn_report(
                    inputs[target[now_indx + arg_offset]], output[now_indx], 
                    label_norm, top_or_loss, multtime, 
                    loss_dict, str_suffix = 'imagenet')

        if instance_task:
            output_offset = 0
            if inst_and_cate:
                output_offset = 1
            if mean_teacher:
                output_offset = 2

            if not inst_cate_sep:
                curr_truth = inputs[target[now_indx + arg_offset]]
            else:
                curr_truth = inputs['label_imagenet_un']

            curr_dist = output[now_indx + output_offset]
            all_labels = output[now_indx + 1 + output_offset]
            if isinstance(all_labels, tuple):
                all_labels = all_labels[0]

            _, top_indices = tf.nn.top_k(curr_dist, k=1)
            curr_pred = tf.gather(all_labels, tf.squeeze(top_indices, axis=1))
            loss_dict['imagenet_top1'] = tf.reduce_mean(
                    tf.cast(
                        tf.equal(curr_pred, tf.cast(curr_truth, tf.int64)), 
                        tf.float32))

            now_indx += 2
            arg_offset -= 2

        if mean_teacher:
            loss_dict = add_topn_report(
                    inputs[target[now_indx + arg_offset]], 
                    output[now_indx + 2],
                    label_norm, top_or_loss, multtime, 
                    loss_dict, str_suffix = 'imagenet_ema')
            now_indx += 3
            arg_offset -= 3

            if instance_task:
                now_indx += 2
                arg_offset -= 2

        now_indx = now_indx + 1

        if extra_feat==1:
            # As we don't need normal and depth here, skip
            now_indx = now_indx + 2
            arg_offset = arg_offset - 2

    if cfg_dataset.get('coco', 0)==1:
        curr_loss = get_semantic_loss(
                curr_predict=output[now_indx], 
                curr_truth=inputs[target[now_indx + arg_offset]], 
                need_mask=True, mask_range=0, less_or_large=1)

        loss_dict['loss_instance_coco'] = tf.reduce_mean(curr_loss)
        now_indx = now_indx + 1

    if cfg_dataset.get('place', 0)==1:
        loss_dict = add_topn_report(
                inputs[target[now_indx + arg_offset]], 
                output[now_indx], 
                label_norm, top_or_loss, multtime, 
                loss_dict, str_suffix = 'place')

        now_indx = now_indx + 1
        if extra_feat==1:
            # As we don't need normal and depth here, skip
            now_indx = now_indx + 2
            arg_offset = arg_offset - 2

    if cfg_dataset.get('kinetics', 0)==1:
        loss_dict = add_topn_report(
                inputs[target[now_indx + arg_offset]], output[now_indx], 
                label_norm, top_or_loss, multtime, 
                loss_dict, str_suffix = 'kinetics')

        now_indx = now_indx + 1

    if cfg_dataset.get('nyuv2', 0)==1:
        curr_loss = depth_loss(
                output[now_indx], inputs[target[now_indx + arg_offset]], 
                depthloss=depthloss, depth_norm=depth_norm)
        loss_dict['loss_depth_nyuv2'] = curr_loss
        now_indx = now_indx + 1

    return loss_dict

def encode_var_filter(curr_tensor):
    curr_name = curr_tensor.name
    if curr_name.startswith('encode') and 'weights' in curr_name:
        return True
    else:
        return False

def save_features(
        inputs, 
        outputs, 
        num_to_save, 
        cfg_dataset={}, 
        depth_norm=8000, 
        target=[], 
        normalloss=0, 
        depthnormal=0,
        extra_feat=0,
        **loss_params
        ):
    save_dict = {}
    now_indx = 0

    if cfg_dataset.get('scenenet', 0)==1:
        image_scenenet = inputs['image_scenenet'][:num_to_save]
        save_dict['fea_image_scenenet'] = image_scenenet

        if cfg_dataset.get('scene_normal', 1)==1:
            normal_scenenet = inputs['normal_scenenet'][:num_to_save]

            normal_scenenet_out = outputs[now_indx][:num_to_save]
            if normalloss==0:
                normal_scenenet_out = tf.multiply(normal_scenenet_out, tf.constant(255, dtype=tf.float32))
                normal_scenenet_out = tf.cast(normal_scenenet_out, tf.uint8)
            now_indx = now_indx + 1
            save_dict['fea_normal_scenenet'] = normal_scenenet
            save_dict['out_normal_scenenet'] = normal_scenenet_out

        if cfg_dataset.get('scene_depth', 1)==1:
            depth_scenenet = inputs['depth_scenenet'][:num_to_save]
            depth_scenenet_out = outputs[now_indx][:num_to_save]
            if depthnormal==0:
                depth_scenenet_out = tf.multiply(depth_scenenet_out, tf.constant(depth_norm, dtype=tf.float32))
                depth_scenenet_out = tf.cast(depth_scenenet_out, tf.int32)
            now_indx = now_indx + 1
            save_dict['fea_depth_scenenet'] = depth_scenenet
            save_dict['out_depth_scenenet'] = depth_scenenet_out

        if cfg_dataset.get('scene_instance', 0)==1:
            instance_scenenet = inputs['instance_scenenet'][:num_to_save]
            instance_scenenet_out = outputs[now_indx][:num_to_save]
            instance_scenenet_out = tf.argmax(instance_scenenet_out, axis = 3)
            now_indx = now_indx + 1
            save_dict['fea_instance_scenenet'] = instance_scenenet
            save_dict['out_instance_scenenet'] = instance_scenenet_out

    if cfg_dataset.get('scannet', 0)==1:
        image_scannet = inputs['image_scannet'][:num_to_save]
        depth_scannet = inputs['depth_scannet'][:num_to_save]

        depth_scannet_out = outputs[now_indx][:num_to_save]
        depth_scannet_out = tf.multiply(depth_scannet_out, tf.constant(depth_norm, dtype=tf.float32))
        depth_scannet_out = tf.cast(depth_scannet_out, tf.int32)
        now_indx = now_indx + 1

        save_dict['fea_image_scannet'] = image_scannet
        save_dict['fea_depth_scannet'] = depth_scannet
        save_dict['out_depth_scannet'] = depth_scannet_out

    if cfg_dataset.get('pbrnet', 0)==1:
        image_pbrnet = inputs['image_pbrnet'][:num_to_save]
        save_dict['fea_image_pbrnet'] = image_pbrnet

        if cfg_dataset.get('pbr_normal', 1)==1:
            normal_pbrnet = inputs['normal_pbrnet'][:num_to_save]

            normal_pbrnet_out = outputs[now_indx][:num_to_save]
            if normalloss==0:
                normal_pbrnet_out = tf.multiply(normal_pbrnet_out, tf.constant(255, dtype=tf.float32))
                normal_pbrnet_out = tf.cast(normal_pbrnet_out, tf.uint8)
            now_indx = now_indx + 1
            save_dict['fea_normal_pbrnet'] = normal_pbrnet
            save_dict['out_normal_pbrnet'] = normal_pbrnet_out

        if cfg_dataset.get('pbr_depth', 1)==1:
            depth_pbrnet = inputs['depth_pbrnet'][:num_to_save]
            depth_pbrnet_out = outputs[now_indx][:num_to_save]
            if depthnormal==0:
                depth_pbrnet_out = tf.multiply(depth_pbrnet_out, tf.constant(depth_norm, dtype=tf.float32))
                depth_pbrnet_out = tf.cast(depth_pbrnet_out, tf.int32)
            now_indx = now_indx + 1
            save_dict['fea_depth_pbrnet'] = depth_pbrnet
            save_dict['out_depth_pbrnet'] = depth_pbrnet_out

        if cfg_dataset.get('pbr_instance', 0)==1:
            instance_pbrnet = inputs['instance_pbrnet'][:num_to_save]
            instance_pbrnet_out = outputs[now_indx][:num_to_save]
            instance_pbrnet_out = tf.argmax(instance_pbrnet_out, axis = 3)
            now_indx = now_indx + 1
            save_dict['fea_instance_pbrnet'] = instance_pbrnet
            save_dict['out_instance_pbrnet'] = instance_pbrnet_out

    if cfg_dataset.get('imagenet', 0)==1:
        now_indx = now_indx + 1

        # If extra_feat, save imagenet images, normals, and depths
        if extra_feat==1:
            image_imagenet = tf.cast(inputs['image_imagenet'][:num_to_save], tf.uint8)

            depth_imagenet_out = outputs[now_indx][:num_to_save]
            now_indx = now_indx + 1

            normal_imagenet_out = outputs[now_indx][:num_to_save]
            now_indx = now_indx + 1

            save_dict['fea_image_imagenet'] = image_imagenet
            save_dict['out_depth_imagenet'] = depth_imagenet_out
            save_dict['out_normal_imagenet'] = normal_imagenet_out


    if cfg_dataset.get('coco', 0)==1:
        image_coco = tf.cast(inputs['image_coco'][:num_to_save], tf.uint8)
        instance_coco = inputs['mask_coco'][:num_to_save]
        instance_coco_out = outputs[now_indx][:num_to_save]
        instance_coco_out = tf.argmax(instance_coco_out, axis = 3)
        now_indx = now_indx + 1
        save_dict['fea_image_coco'] = image_coco
        save_dict['fea_instance_coco'] = instance_coco
        save_dict['out_instance_coco'] = instance_coco_out

    if cfg_dataset.get('place', 0)==1:
        now_indx = now_indx + 1

        # If extra_feat, save place images, normals, and depths
        if extra_feat==1:
            image_place = tf.cast(inputs['image_place'][:num_to_save], tf.uint8)

            depth_place_out = outputs[now_indx][:num_to_save]
            now_indx = now_indx + 1

            normal_place_out = outputs[now_indx][:num_to_save]
            now_indx = now_indx + 1

            save_dict['fea_image_place'] = image_place
            save_dict['out_depth_place'] = depth_place_out
            save_dict['out_normal_place'] = normal_place_out

    if cfg_dataset.get('nyuv2', 0)==1:
        image_nyuv2 = inputs['image_nyuv2'][:num_to_save]
        depth_nyuv2 = inputs['depth_nyuv2'][:num_to_save]

        depth_nyuv2_out = outputs[now_indx][:num_to_save]
        if depthnormal==0:
            depth_nyuv2_out = tf.multiply(
                    depth_nyuv2_out, 
                    tf.constant(depth_norm, dtype=tf.float32))
            depth_nyuv2_out = tf.cast(depth_nyuv2_out, tf.int32)
        now_indx = now_indx + 1

        save_dict['fea_image_nyuv2'] = image_nyuv2
        save_dict['fea_depth_nyuv2'] = depth_nyuv2
        save_dict['out_depth_nyuv2'] = depth_nyuv2_out

    return save_dict

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

def gpu_col_feat(inputs, outputs, *args):
    #print("gpu_col_feat inputs:", inputs)
    #print("gpu_col_feat outputs:", outputs)
    soft=True
    num_images = 1
    l_image = outputs['l'][:num_images] + 50
    pred_ab_image = Q_to_ab(outputs['logits'][:num_images], soft=soft)
    pred_ab_image = tf.image.resize_images(pred_ab_image, l_image.get_shape().as_list()[1:3])
    pred_lab_image = tf.concat([l_image, pred_ab_image], axis=-1)
    pred_image = lab_to_rgb(pred_lab_image)
    pred_image = tf.multiply(pred_image, 255.0)

    original_ab_image = Q_to_ab(outputs['Q'][:num_images], soft=soft, is_logits=False)
    original_ab_image = tf.image.resize_images(original_ab_image, l_image.get_shape().as_list()[1:3])
    original_lab_image = tf.concat([l_image, original_ab_image], axis=-1)
    original = lab_to_rgb(original_lab_image)
    original = tf.multiply(original, 255.0)

    raw_ab_image = outputs['ab_image'][:num_images]
    raw_lab_image = tf.concat([l_image, raw_ab_image], axis=-1)
    raw_original = lab_to_rgb(raw_lab_image)
    raw_original = tf.multiply(raw_original, 255.0)

    return {'pred': pred_image[:num_images],
            'original': original[:num_images],
            'raw': inputs['image_imagenet'][:num_images],
            'raw_original': raw_original[:num_images],
            }
