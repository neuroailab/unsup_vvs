from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys
import copy
import pdb

from nf_utils import rgb_to_lab
sys.path.append('../combine_pred/')
from combinet_builder import build_partnet, getVarName, getFdbVarName
from models.instance_task.model.instance_model import resnet_embedding
from models.instance_task.model.resnet_th_preprocessing import ApplySobel
from models.model_blocks import NoramlNetfromConv

MEAN_RGB = [0.485, 0.456, 0.406]


def spatial_select_out_layers(
        all_out_dict_dataset, 
        v4_nodes, it_nodes, 
        spatial_select):
    all_keys = all_out_dict_dataset.keys()
    for curr_key in all_keys:
        if not curr_key in (v4_nodes + it_nodes):
            continue
        curr_layer = all_out_dict_dataset[curr_key]
        curr_shape = curr_layer.get_shape().as_list()
        assert len(curr_shape) == 4
        # Allowing both c, x, y and x, y, c
        spatial_size = curr_shape[2]
        c_last = curr_shape[2] == curr_shape[1]
        stride, size = spatial_select.split(',')
        stride = int(stride)
        size = int(size)
        spatial_upper = spatial_size-size+1
        all_start_idxs = range(0, spatial_upper, stride)
        if all_start_idxs[-1] + size < spatial_size:
            all_start_idxs.append(spatial_size - size)
        for x_idx, x_pos in enumerate(all_start_idxs):
            for y_idx, y_pos in enumerate(all_start_idxs):
                if c_last:
                    curr_sub_layer = curr_layer[:, \
                                                x_pos:x_pos+size, \
                                                y_pos:y_pos+size, :]
                else:
                    curr_sub_layer = curr_layer[:, :, \
                                                x_pos:x_pos+size, \
                                                y_pos:y_pos+size]
                new_name = curr_key + ('_%i_%i' % (x_idx, y_idx))
                all_out_dict_dataset[new_name] = curr_sub_layer

    return all_out_dict_dataset


def combinet_neural_fit(
        inputs,
        cfg_initial,
        no_prep=0, 
        center_im=1,
        v4_nodes=None,
        it_nodes=None,
        v4_out_shape=88,
        it_out_shape=168,
        cache_filter=1,
        train=True,
        init_type='xavier',
        weight_decay=None, 
        weight_decay_type='l2', 
        random_gather=False,
        random_sample=None,
        random_sample_seed=None,
        in_conv_form=False,
        ignorebname_new=1,
        batch_name='_imagenet',
        f10ms_time=None,
        partnet_train=False,
        gen_features=False,
        use_precompute=False,
        color_prep=0,
        combine_col_rp=0,
        rp_sub_mean=0,
        div_std=0,
        input_mode='rgb',
        mean_teacher=False,
        ema_decay=0.9997,
        ema_zerodb=False,
        inst_model=None,
        inst_res_size=18,
        spatial_select=None,
        convrnn_model=False,
        **kwargs
        ):
    if mean_teacher:
        kwargs['fixweights'] = False
        kwargs['sm_bn_trainable'] = True

    output_nodes = {}
    if not use_precompute:
        dataset_prefix_list = [
                'scenenet', 'pbrnet', 'imagenet', 'coco', 
                'place', 'rp', 'colorization', 'rp_imagenet']
        combined_orders = []
        for which_dataset in dataset_prefix_list: 
            curr_order = cfg_initial.get('%s_order' % which_dataset, [])
            for which_partnet in curr_order:
                if which_partnet not in combined_orders:
                    combined_orders.append(which_partnet)
        print(combined_orders)

        if len(combined_orders)==0:
            combined_orders = ['encode', 'category']
        if 'ae_output' in combined_orders:
            combined_orders = ['encode']

        # Prepare the input
        image_dataset = tf.cast(inputs['images'], tf.float32)
        image_shape = image_dataset.get_shape().as_list()
        if image_shape[1] != 224:
            print('Image shape different from 224!', image_shape)
            print('Excluding category from orders!')
            if 'category' in combined_orders:
                combined_orders.remove('category')
            if 'inst_category' in combined_orders:
                combined_orders.remove('inst_category')
            if 'memory' in combined_orders:
                combined_orders.remove('memory')

        if rp_sub_mean==1:
            offset = tf.constant(MEAN_RGB, shape=[1, 1, 1, 3])
            image_dataset -= offset
        if div_std==1:
            STDDEV_RGB = [0.229, 0.224, 0.225]
            scale = tf.constant(STDDEV_RGB, shape=[1, 1, 1, 3])
            image_dataset /= scale
        if color_prep==1:
            print("*****************color_prep*****************")
            image_dataset = rgb_to_lab(image_dataset)
            image_dataset = image_dataset[:,:,:,:1] - 50
            if combine_col_rp==1:
                image_dataset = tf.tile(image_dataset, [1, 1, 1, 3])
        if no_prep==0:
            image_dataset = tf.div(
                    image_dataset, 
                    tf.constant(255, dtype=tf.float32))
            if center_im:
                image_dataset  = tf.subtract(
                        image_dataset, 
                        tf.constant(0.5, dtype=tf.float32))
        if input_mode == 'sobel':
            image_dataset = ApplySobel(image_dataset)
            image_dataset = tf.squeeze(image_dataset)

        # Initialize dictionary storing all outputs
        all_out_dict_dataset = {}

        def _build_network(all_out_dict_dataset):
            # Actually build the network
            dict_cache_filter = {}
            reuse_dict = {}
            first_flag = True
            for network_name in combined_orders:
                if first_flag:
                    input_now = image_dataset
                    first_flag = False
                else:
                    input_now = None

                var_name = getVarName(cfg_initial, key_want = network_name)
                reuse_name = '%s_reuse' % var_name
                reuse_curr = reuse_dict.get(reuse_name, None)

                fdb_var_name = getFdbVarName(
                        cfg_initial, 
                        key_want = network_name)
                fdb_reuse_name = '_fdb_%s_reuse' % fdb_var_name
                fdb_reuse_curr = reuse_dict.get(fdb_reuse_name, None)

                _, all_out_dict_dataset = build_partnet(
                        input_now, 
                        cfg_initial=cfg_initial, 
                        key_want=network_name, 
                        reuse_flag=reuse_curr, 
                        fdb_reuse_flag=fdb_reuse_curr, 
                        #reuse_batch=None, 
                        reuse_batch=reuse_curr, 
                        #batch_name='_%s' % network_name, 
                        batch_name=batch_name, 
                        all_out_dict=all_out_dict_dataset, 
                        cache_filter=cache_filter, 
                        init_type=init_type,
                        dict_cache_filter=dict_cache_filter,
                        weight_decay=None, 
                        train=partnet_train,
                        ignorebname_new=ignorebname_new,
                        **kwargs)

                reuse_dict[reuse_name] = True
                reuse_dict[fdb_reuse_name] = True

            return all_out_dict_dataset

        if not mean_teacher:
            if inst_model:
                inst_input = inputs['images']
                all_out_dict_dataset = resnet_embedding(
                        inst_input * 255,
                        get_all_layers=inst_model,
                        skip_final_dense=True,
                        input_mode=input_mode,
                        resnet_size=inst_res_size,
                        )
            elif convrnn_model:
                from models.convrnn_model import convrnn_model_func
                all_out_dict_dataset = convrnn_model_func(inputs)
            else:
                all_out_dict_dataset = _build_network(all_out_dict_dataset)
        else:
            # Build two networks (primary and ema)
            primary_out_dict = {}
            ema_out_dict = {}

            # Build the model in the special name scope
            with name_variable_scope("primary", "primary", reuse=tf.AUTO_REUSE) as (name_scope, var_scope):
                primary_out_dict = _build_network(primary_out_dict)

            # Build the teacher model using ema_variable_scope
            with ema_variable_scope("ema", var_scope, decay=ema_decay, zero_debias=ema_zerodb, reuse=tf.AUTO_REUSE):
                ema_out_dict = _build_network(ema_out_dict)

            # Combine two out dicts, adding $ema_ as prefix to keys in ema_out_dict
            for each_key, each_value in ema_out_dict.items():
                new_key = "ema_%s" % each_key
                assert not new_key in primary_out_dict, "New key %s already exists in original network" % new_key
                primary_out_dict[new_key] = each_value
            all_out_dict_dataset = primary_out_dict

    assert (v4_nodes is not None) or (it_nodes is not None), 'Must set some fitting nodes'
    if spatial_select:
        all_out_dict_dataset = spatial_select_out_layers(
                all_out_dict_dataset, v4_nodes, it_nodes, spatial_select)

    m_fit = NoramlNetfromConv(**kwargs)

    random_seed_offset = 0

    def _add_pred_layer(curr_in_layer, out_shape, random_seed_offset):
        if random_gather:
            curr_output = m_fit.random_gather(
                    in_layer=curr_in_layer,
                    random_sample=random_sample,
                    random_sample_seed=random_sample_seed+random_seed_offset,
                    )
            random_seed_offset = random_seed_offset + 1

        elif len(curr_in_layer.get_shape().as_list())==4:
            curr_output = m_fit.spa_disen_fc(
                    out_shape,
                    in_layer=curr_in_layer,
                    bias=0,
                    trainable=True,
                    weight_decay=weight_decay,
                    weight_decay_type=weight_decay_type,
                    in_conv_form=in_conv_form,
                    )
        else:
            curr_output = m_fit.fc(
                    out_shape,
                    in_layer=curr_in_layer,
                    bias=0,
                    trainable=True,
                    weight_decay=weight_decay,
                    )
        return curr_output, random_seed_offset

    def _get_input_layer(all_out_dict_dataset, each_node):
        if not ':' in each_node:
            assert each_node in all_out_dict_dataset, 'Node %s not constructed yet!' % each_node

            curr_in_layer = all_out_dict_dataset[each_node]
            each_node_new = each_node
        else:
            each_node_list = each_node.split(':')
            curr_in_layers = []
            ds_now = None
            for x_node in each_node_list:
                assert x_node in all_out_dict_dataset, 'Node %s not constructed yet!' % x_node
                curr_in_layer = all_out_dict_dataset[x_node]
                if ds_now is None:
                    ds_now = curr_in_layer.get_shape().as_list()[1]
                curr_ds = curr_in_layer.get_shape().as_list()[1]
                if not ds_now==curr_ds:
                    curr_in_layer = tf.image.resize_images(curr_in_layer, [ds_now, ds_now])
                    print('Resize %s for %s' % (x_node, each_node_list))
                curr_in_layers.append(curr_in_layer)
            curr_in_layer = tf.concat(curr_in_layers, axis=-1)
            each_node_new = each_node.replace(':','-')

        return curr_in_layer,each_node_new

    with tf.variable_scope('v4_fit'):
        if v4_nodes is not None:
            v4_node_list = v4_nodes.split(',')
            print('%s fit to V4 responses' % v4_nodes)
            for each_node in v4_node_list:
                if not use_precompute: 
                    curr_in_layer,each_node_new = _get_input_layer(all_out_dict_dataset, each_node)
                else:
                    each_node_new = each_node.replace(':', '-')
                    assert each_node_new in inputs, 'Input %s not found in inputs' % each_node_new
                    curr_in_layer = inputs[each_node_new]

                if gen_features:
                    # If the network is built for generating the features,
                    # then just put the curr_in_layer as output
                    output_nodes[each_node] = curr_in_layer
                    continue

                if f10ms_time is None:
                    with tf.variable_scope('%s_fit' % each_node_new):
                        curr_output, random_seed_offset = _add_pred_layer(curr_in_layer, v4_out_shape, random_seed_offset)
                    if random_gather:
                        output_nodes[each_node] = curr_output
                    else:
                        output_nodes['v4/%s' % each_node] = curr_output
                else:
                    all_time_steps = f10ms_time.split(',')
                    for curr_time_step in all_time_steps:
                        with tf.variable_scope('%s_fit_%s0ms' % (each_node_new, curr_time_step)):
                            curr_output, random_seed_offset = _add_pred_layer(curr_in_layer, v4_out_shape, random_seed_offset)
                        if random_gather:
                            output_nodes[each_node] = curr_output
                        else:
                            output_nodes['v4_%s/%s' % (curr_time_step, each_node)] = curr_output

    #random_seed_offset = 0
    with tf.variable_scope('it_fit'):
        if it_nodes is not None:
            it_node_list = it_nodes.split(',')
            print('%s fit to IT responses' % it_nodes)
            for each_node in it_node_list:
                if not use_precompute: 
                    curr_in_layer,each_node_new = _get_input_layer(all_out_dict_dataset, each_node)
                else:
                    each_node_new = each_node.replace(':', '-')
                    assert each_node_new in inputs, 'Input %s not found in inputs' % each_node_new
                    curr_in_layer = inputs[each_node_new]

                if gen_features:
                    if each_node not in output_nodes:
                        output_nodes[each_node] = curr_in_layer
                    continue

                if f10ms_time is None:
                    with tf.variable_scope('%s_fit' % each_node_new):
                        curr_output, random_seed_offset = _add_pred_layer(curr_in_layer, it_out_shape, random_seed_offset)
                    if random_gather:
                        if each_node not in output_nodes:
                            output_nodes[each_node] = curr_output
                    else:
                        output_nodes['it/%s' % each_node] = curr_output
                else:
                    all_time_steps = f10ms_time.split(',')
                    for curr_time_step in all_time_steps:
                        with tf.variable_scope('%s_fit_%s0ms' % (each_node_new, curr_time_step)):
                            curr_output, random_seed_offset = _add_pred_layer(curr_in_layer, it_out_shape, random_seed_offset)
                        if random_gather:
                            if each_node not in output_nodes:
                                output_nodes[each_node] = curr_output
                        else:
                            output_nodes['it_%s/%s' % (curr_time_step, each_node)] = curr_output

    ret_params = m_fit.params
    print("Builder Done!")

    # remove all other parameters
    all_train_ref = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    cp_all_train_ref = copy.copy(all_train_ref)
    for each_v in cp_all_train_ref:
        if 'it_fit' in each_v.op.name or 'v4_fit' in each_v.op.name:
            continue
        else:
            all_train_ref.remove(each_v)
    print("Trainable vars:")
    print([
        v.op.name \
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
    # remove all update ops
    all_update_ref = tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)
    cp_all_update_ref = copy.copy(all_update_ref)
    for each_v in cp_all_update_ref:
        all_update_ref.remove(each_v)
    print("UPDATE vars:")
    print([
        v.op.name \
        for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)])
    return output_nodes, ret_params
