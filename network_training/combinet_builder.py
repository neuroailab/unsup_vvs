from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys
import copy
import pdb

from resnet_th_preprocessing import ColorNormalize
from utils import pos2lbl, rgb_to_lab
from utils import ema_variable_scope, name_variable_scope

sys.path.append('../normal_pred/')
from normal_encoder_asymmetric_with_bypass import *
from resnet_preprocessing import col_preprocess_for_gpu

def getWhetherResBlock(i, cfg, key_want="encode"):
    tmp_dict = cfg[key_want][i]
    return 'ResBlock' in tmp_dict


# sm_add
def getWhetherUpProj(i, cfg, key_want="up_projection"):
    tmp_dict = cfg[key_want][i]
    return 'UpProj' in tmp_dict


def getBypassAdd(i, cfg, key_want="encode"):
    #print("****************key_want:", key_want)
    #print(i)
    tmp_dict = cfg[key_want][i]
    ret_val = tmp_dict.get('bypass_add', None)
    return ret_val


def getResBlockSettings(i, cfg, key_want="encode"):
    tmp_dict = cfg[key_want][i]
    return tmp_dict['ResBlock']


# sm_add
def getUpProjSettings(i, cfg, key_want="up_projection"):
    tmp_dict = cfg[key_want][i]
    return tmp_dict['UpProj']


def getWhetherBn(i, cfg, key_want="encode"):
    tmp_dict = cfg[key_want][i]
    return 'bn' in tmp_dict

def getWhetherSoftmax(i, cfg, key_want="encode"):
    tmp_dict = cfg[key_want][i]
    return 'softmax' in tmp_dict


def getWhetherKin(cfg, key_want="encode"):
    tmp_dict = cfg[key_want]
    return 'kin_act' in tmp_dict


def getKinFrom(cfg, key_want="encode"):
    tmp_dict = cfg[key_want]
    return tmp_dict['kin_act']


def getKinSplitFrom(cfg, key_want="encode"):
    tmp_dict = cfg[key_want]
    return tmp_dict['split_para']


def getWhetherFdb(i, cfg, key_want="encode"):
    tmp_dict = cfg[key_want][i]
    return 'fdb' in tmp_dict


def getFdbFrom(i, cfg, key_want="encode"):
    tmp_dict = cfg[key_want][i]['fdb']
    return tmp_dict['from']


def getFdbType(i, cfg, key_want="encode"):
    tmp_dict = cfg[key_want][i]['fdb']
    return tmp_dict['type']


def getDepConvWhetherBn(i, cfg, key_want="encode"):
    val = False
    tmp_dict = cfg[key_want][i]
    if 'conv' in tmp_dict:
        val = 'bn' in tmp_dict['conv']
    return val


def getConvOutput(i, cfg, key_want="encode"):
    tmp_dict = cfg[key_want][i]["conv"]
    return tmp_dict.get("output", 0) == 1


def getWhetherInitFile(i, cfg, key_want="encode", layer_type="conv"):
    tmp_dict = cfg[key_want][i][layer_type]
    return "init_file" in tmp_dict


def getInitFileName(i, cfg, key_want="encode", layer_type="conv"):
    tmp_dict = cfg[key_want][i][layer_type]
    init_path = tmp_dict["init_file"]
    if init_path[0] == '$':
        init_path = cfg[init_path[1:]]
    return init_path


def getInitFileArgs(i, cfg, key_want="encode", layer_type="conv"):
    tmp_dict = cfg[key_want][i][layer_type]
    init_args = tmp_dict["init_layer_keys"]
    return init_args


def getVarName(cfg, key_want="encode"):
    tmp_dict = cfg[key_want]
    return tmp_dict.get('var_name', key_want)


def getBnVarName(cfg, key_want="encode"):
    tmp_dict = cfg[key_want]
    return tmp_dict.get('bn_var_name', '')


def getVarOffset(cfg, key_want="encode"):
    tmp_dict = cfg[key_want]
    return tmp_dict.get('var_offset', 0)


def getFdbVarName(cfg, key_want="encode"):
    tmp_dict = cfg[key_want]
    return tmp_dict.get('fdb_var_name', key_want)


def getFdbVarOffset(cfg, key_want="encode"):
    tmp_dict = cfg[key_want]
    return tmp_dict.get('fdb_var_offset', 0)


def getEncodeConvBn(i, cfg, which_one='encode'):
    val = False

    if which_one in cfg and (i in cfg[which_one]):
        if 'conv' in cfg[which_one][i]:
            if 'bn' in cfg[which_one][i]['conv']:
                if cfg[which_one][i]['conv']['bn'] == 1:
                    val = True

    return val


def getPoolPadding(i, cfg, which_one='encode'):
    val = 'SAME'

    if which_one in cfg and (i in cfg[which_one]):
        if 'pool' in cfg[which_one][i]:
            if 'padding' in cfg[which_one][i]['pool']:
                val = cfg[which_one][i]['pool']['padding']

    return val


def getConvPadding(i, cfg, which_one='encode'):
    val = None

    if which_one in cfg and (i in cfg[which_one]):
        if 'conv' in cfg[which_one][i]:
            if 'padding' in cfg[which_one][i]['conv']:
                val = cfg[which_one][i]['conv']['padding']

    return val

def getConvDilat(i, cfg, which_one='encode'):
    val = 1

    if which_one in cfg and (i in cfg[which_one]):
        if 'conv' in cfg[which_one][i]:
            if 'dilat' in cfg[which_one][i]['conv']:
                val = cfg[which_one][i]['conv']['dilat']

    return val

def getConvUpsample(i, cfg, which_one='encode'):
    val = None

    if which_one in cfg and (i in cfg[which_one]):
        if 'conv' in cfg[which_one][i]:
            if 'upsample' in cfg[which_one][i]['conv']:
                val = cfg[which_one][i]['conv']['upsample']

    return val


# Function for building subnetwork based on configurations
def build_partnet(
        inputs,
        cfg_initial,
        key_want='encode',
        train=True,
        seed=None,
        reuse_flag=None,
        reuse_batch=None,
        fdb_reuse_flag=None,
        batch_name='',
        all_out_dict={},
        init_stddev=.01,
        ignorebname=0,
        ignorebname_new=1,
        weight_decay=None,
        init_type='xavier',
        cache_filter=0,
        dict_cache_filter={},
        fix_pretrain=0,
        corr_bypassadd=0,
        sm_fix=0,  # sm_add
        sm_de_fix=0,  # sm_add
        sm_depth_fix=0,  # sm_add
        sm_resnetv2=0,
        sm_resnetv2_1=0,
        sm_bn_trainable=True,
        sm_bn_fix=0,
        tpu_flag=0,
        tpu_depth=0,
        train_anyway=0,
        combine_fewshot=0,
        **kwargs):
    cfg = cfg_initial
    if seed == None:
        fseed = getFilterSeed(cfg)
    else:
        fseed = seed

    batch_name_new = batch_name
    reuse_batch_new = reuse_batch

    if ignorebname == 1:
        batch_name = ''
        reuse_batch = reuse_flag

    if ignorebname_new == 1:
        batch_name_new = ''
        reuse_batch_new = reuse_flag

    m = NoramlNetfromConv(seed=fseed, **kwargs)

    # sm_add
    sm_freeze = -1
    if key_want == 'encode' and sm_fix >= 1:
        print("Fix_Encode_Path!")
        sm_freeze = sm_fix - 1
    
    sm_bn_freeze = -1
    if key_want == 'encode' and sm_bn_fix >= 1:
        sm_bn_freeze = sm_bn_fix - 1

    sm_decode_freeze = -1
    if key_want == 'decode' and sm_de_fix > 0:
        print("Fix_Decode_Path!")
        sm_decode_freeze = sm_de_fix - 1

    sm_depth_freeze = -1
    if key_want == 'depth' and sm_depth_fix > 0:
        print("Fix_Depth_Path!")
        sm_depth_freeze = sm_depth_fix - 1

    if tpu_depth == 1:
        train=True

    if train_anyway == 1:
        train=True

    assert key_want in cfg, "Wrong key %s for network" % key_want

    valid_flag = True
    if inputs is None:
        print("key want:", key_want)
        assert 'input' in cfg[key_want], "No inputs specified for network %s!" % key_want
        input_node = cfg[key_want]['input']
        print(input_node)
        assert input_node in all_out_dict, "Input nodes not built yet for network %s!" % key_want
        inputs = all_out_dict[input_node]
        valid_flag = False
    if getWhetherKin(cfg_initial, key_want=key_want):

        # Action related for kinetics

        kin_act = getKinFrom(cfg, key_want=key_want)

        # Reshape: put the time dimension to channel directly, assume time dimension is second dimension
        if kin_act == 'reshape':
            inputs = tf.transpose(inputs, perm=[0, 2, 3, 4, 1])
            curr_shape = inputs.get_shape().as_list()
            inputs = tf.reshape(inputs, [curr_shape[0], curr_shape[1], curr_shape[2], -1])

        # Split: split the time dimension, build shared networks for all splits
        if kin_act == 'split':
            split_para = getKinSplitFrom(cfg, key_want=key_want)
            split_inputs = tf.split(inputs, num_or_size_splits=split_para, axis=1)

            new_cfg = copy.deepcopy(cfg)
            new_cfg[key_want]['kin_act'] = 'reshape'
            add_out_dict = {}
            all_outs = []

            for split_indx, curr_split in enumerate(split_inputs):
                curr_all_out_dict = copy.copy(all_out_dict)
                curr_m, curr_all_out_dict = build_partnet(
                    curr_split, new_cfg, key_want=key_want, train=train,
                    seed=seed, reuse_flag=reuse_flag or (split_indx > 0),
                    reuse_batch=reuse_batch or (split_indx > 0),
                    fdb_reuse_flag=fdb_reuse_flag or (split_indx > 0),
                    batch_name=batch_name, all_out_dict=curr_all_out_dict,
                    init_stddev=init_stddev, ignorebname=ignorebname,
                    weight_decay=weight_decay, cache_filter=cache_filter,
                    dict_cache_filter=dict_cache_filter,
                    **kwargs)
                all_outs.append(curr_m.output)
                for layer_name in curr_all_out_dict:
                    if layer_name in all_out_dict:
                        continue
                    if not layer_name in add_out_dict:
                        add_out_dict[layer_name] = []
                    add_out_dict[layer_name].append(curr_all_out_dict[layer_name])

            for layer_name in add_out_dict:
                all_out_dict[layer_name] = tf.stack(add_out_dict[layer_name], axis=1)

            curr_m.output = tf.stack(all_outs, axis=1)

            return curr_m, all_out_dict

    # Set the input
    m.output = inputs

    # General network building
    with tf.contrib.framework.arg_scope([m.conv], init=init_type,
                                        stddev=init_stddev, bias=0, activation='relu'):
        encode_depth = getPartnetDepth(cfg, key_want=key_want)

        
        # Sometimes we want this network share parameters with different network
        # we can achieve that by setting var_name (variable name) and var_offset (layer offset for sharing)
        var_name = getVarName(cfg, key_want=key_want)
        var_offset = getVarOffset(cfg, key_want=key_want)

        # fdb connections may have different var_name and var_offset
        fdb_var_name = getFdbVarName(cfg, key_want=key_want)
        fdb_var_offset = getFdbVarOffset(cfg, key_want=key_want)

        # Build each layer, as cfg file starts from 1, we also start from 1
        for i in range(1, encode_depth + 1):
            layer_name = "%s_%i" % (key_want, i)
            curr_reuse_name = '%s%i' % (var_name, i + var_offset)

            # Build addition bypass
            add_bypass_add = getBypassAdd(i, cfg, key_want=key_want)
            if add_bypass_add is not None:
                print("add_bypass_add appear")
                for bypass_add_layer_name in add_bypass_add:
                    assert bypass_add_layer_name in all_out_dict, "Node %s not built yet for network %s!" % (
                    bypass_add_layer_name, key_want)
                    bypass_add_layer = all_out_dict[bypass_add_layer_name]
                    curr_output = m.output
                    nf_output = curr_output.get_shape().as_list()[-1]
                    bypass_add_layer_after = m.conv(nf_output, 1, 1, activation=None, bias=0, padding='SAME',
                                                    in_layer=bypass_add_layer,
                                                    weight_decay=weight_decay, init=init_type,
                                                    train=train,
                                                    whetherBn=True,
                                                    reuse_name=curr_reuse_name,
                                                    reuse_flag=reuse_flag,
                                                    batch_name=batch_name_new,
                                                    batch_reuse=reuse_batch_new,
                                                    )
                    if corr_bypassadd == 0:
                        m.output = curr_output
                        new_encode_node = m.add_bypass(bypass_add_layer)
                    else:
                        m.output = curr_output
                        new_encode_node = m.add_bypass_adding(bypass_add_layer_after)

            with tf.variable_scope(curr_reuse_name, reuse=reuse_flag):
                # add bypass
                add_bypass = getDecodeBypass_light(i, cfg, key_want=key_want)

                if add_bypass != None:
                    for bypass_layer_name in add_bypass:
                        if bypass_layer_name == '_coord':
                            new_encode_node = m.add_coord()
                            # print('Add Coord here!')
                            continue

                        assert bypass_layer_name in all_out_dict, "Node %s not built yet for network %s!" % (
                            bypass_layer_name, key_want)
                        bypass_layer = all_out_dict[bypass_layer_name]
                        new_encode_node = m.add_bypass(bypass_layer)
                        #print("*********")
                        #print(new_encode_node.shape)
                        # print('Network %s bypass from %s at %s' % (key_want, bypass_layer_name, layer_name))

            with tf.variable_scope(curr_reuse_name):
                # Build resnet block
                if getWhetherResBlock(i, cfg, key_want=key_want):
                    # sm_add
                    sm_trainable = None
                    if sm_freeze == -1:
                        sm_trainable = None
                    elif encode_depth - i >= sm_freeze:
                        sm_trainable = False
                    
                    #sm_bn_trainable = True
                    if sm_bn_freeze == -1:
                        sm_bn_trainable = sm_bn_trainable
                    else:
                        if encode_depth - i >= sm_bn_freeze:
                            sm_bn_trainable = False
                        else:
                            sm_bn_trainable = True
                    # print('resblock_%i' % i)



                    if sm_resnetv2 == 0 and sm_resnetv2_1 == 0:
                        #print("is_training:", train)
                        new_encode_node = m.resblock(
                            conv_settings=getResBlockSettings(i, cfg, key_want=key_want),
                            weight_decay=weight_decay, bias=0, init=init_type,
                            stddev=init_stddev, train=train, padding='SAME',
                            reuse_flag=reuse_flag,
                            batch_name=batch_name_new,
                            batch_reuse=reuse_batch_new,
                            sm_trainable=sm_trainable,
                            sm_bn_trainable=sm_bn_trainable,
                            tpu_flag=tpu_flag,
                            combine_fewshot=combine_fewshot,
                        )
                    elif sm_resnetv2 == 1:
                        new_encode_node = m.resblockv2(
                            conv_settings=getResBlockSettings(i, cfg, key_want=key_want),
                            weight_decay=weight_decay, bias=0, init=init_type,
                            stddev=init_stddev, train=train, padding='SAME',
                            reuse_flag=reuse_flag,
                            batch_name=batch_name_new,
                            batch_reuse=reuse_batch_new,
                            sm_trainable=sm_trainable,
                            sm_bn_trainable=sm_bn_trainable,
                            tpu_flag=tpu_flag,
                        )
                    else:
                        new_encode_node = m.resblockv2_1(
                            conv_settings=getResBlockSettings(i, cfg, key_want=key_want),
                            weight_decay=weight_decay, bias=0, init=init_type,
                            stddev=init_stddev, train=train, padding='SAME',
                            reuse_flag=reuse_flag,
                            batch_name=batch_name_new,
                            batch_reuse=reuse_batch_new,
                            sm_trainable=sm_trainable,
                            sm_bn_trainable=sm_bn_trainable,
                        )

            # do convolution
            if getDoConv(i, cfg, which_one=key_want):
                cfs = getEncodeConvFilterSize(i, cfg, which_one=key_want)
                nf = getEncodeConvNumFilters(i, cfg, which_one=key_want)
                cs = getEncodeConvStride(i, encode_depth, cfg, which_one=key_want)
                cvBn = getEncodeConvBn(i, cfg, which_one=key_want)
                #print("cvBn:", cvBn)
                conv_padding = getConvPadding(i, cfg, which_one=key_want)
                dilat = getConvDilat(i, cfg, which_one=key_want)
                if combine_fewshot ==1:
                    dilat = 1

                trans_out_shape = None
                conv_upsample = getConvUpsample(i, cfg, which_one=key_want)
                if not conv_upsample is None:
                    trans_out_shape = m.output.get_shape().as_list()
                    trans_out_shape[1] = conv_upsample * trans_out_shape[1]
                    trans_out_shape[2] = conv_upsample * trans_out_shape[2]
                    trans_out_shape[3] = nf

                padding = 'SAME'
                activation = 'relu'
                bias = 0

                if valid_flag:
                    padding = 'VALID'
                    valid_flag = False
                else:
                    if getConvOutput(i, cfg, key_want=key_want):
                        activation = None
                        bias = 0

                if conv_padding != None:
                    padding = conv_padding

                init = init_type
                init_file = None
                init_layer_keys = None

                trainable = None

                # sm_add
                if sm_freeze != -1 and key_want == 'encode':
                    trainable = False
                if sm_decode_freeze != -1 and key_want == 'decode':
                    trainable = False
                if sm_depth_freeze != -1 and key_want == 'depth':
                    trainable = False
                if sm_bn_freeze != -1 and key_want == 'encode':
                    sm_bn_trainable = False

                # if getWhetherInitFile(i, cfg, key_want = key_want) and (reuse_flag!=True):
                if getWhetherInitFile(i, cfg, key_want=key_want):
                    init = 'from_file'
                    init_file = getInitFileName(i, cfg, key_want=key_want)
                    init_layer_keys = getInitFileArgs(i, cfg, key_want=key_want)

                    # if cache_filter is 1, will load into a tensor, save it for later reuse
                    if cache_filter == 1:
                        init = 'from_cached'
                        filter_cache_str_prefix = '%s_%i' % (var_name, i + var_offset)
                        weight_key = '%s/weight' % filter_cache_str_prefix
                        bias_key = '%s/bias' % filter_cache_str_prefix
                        if not weight_key in dict_cache_filter:
                            params = np.load(init_file)
                            dict_cache_filter[weight_key] = tf.constant(params[init_layer_keys['weight']],
                                                                        dtype=tf.float32)
                            dict_cache_filter[bias_key] = tf.constant(params[init_layer_keys['bias']], dtype=tf.float32)

                        init_layer_keys = {'weight': dict_cache_filter[weight_key], 'bias': dict_cache_filter[bias_key]}
                    else:
                        print('Layer conv %s init from file' % layer_name)

                    if fix_pretrain == 1:
                        trainable = False

                if not getConvDepsep(i, cfg, which_one=key_want):
                    new_encode_node = m.conv(nf, cfs, cs, activation=activation, bias=bias, padding=padding,
                                             weight_decay=weight_decay, init=init, init_file=init_file, whetherBn=cvBn,
                                             train=train, init_layer_keys=init_layer_keys,
                                             trans_out_shape=trans_out_shape,
                                             trainable=trainable,
                                             reuse_name=curr_reuse_name,
                                             reuse_flag=reuse_flag,
                                             batch_name=batch_name_new,
                                             batch_reuse=reuse_batch_new,
                                             sm_bn_trainable=sm_bn_trainable,
                                             tpu_flag=tpu_flag,
                                             dilat=dilat,
                                             )
                else:
                    with tf.variable_scope(curr_reuse_name, reuse=reuse_flag):
                        with_bn = getDepConvWhetherBn(i, cfg, key_want=key_want)
                        new_encode_node = m.depthsep_conv(nf, getConvDepmul(i, cfg, which_one=key_want), cfs, cs,
                                                          dep_padding=padding, sep_padding=padding,
                                                          activation=activation, bias=bias,
                                                          with_bn=with_bn, bn_name=batch_name, reuse_batch=reuse_batch,
                                                          train=train,
                                                          weight_decay=weight_decay,
                                                          sm_bn_trainable=sm_bn_trainable,
                                                          tpu_flag=tpu_flag,
                                                          )

                        # print('Network %s conv %s with size %d stride %d numfilters %d' % (key_want, layer_name, cfs, cs, nf))

            with tf.variable_scope(curr_reuse_name, reuse=reuse_flag):

                # do unpool
                do_unpool = getDecodeDoUnPool(i, cfg, key_want=key_want)
                if do_unpool:
                    unpool_scale = getDecodeUnPoolScale(i, cfg, key_want=key_want)
                    new_encode_node = m.resize_images_scale(unpool_scale)

                    # print('Network %s unpool %s with scale %d' % (key_want, layer_name, unpool_scale))

                if getDoFc(i, cfg, which_one=key_want):

                    init = 'trunc_norm'
                    init_file = None
                    init_layer_keys = None

                    if getWhetherInitFile(i, cfg, key_want=key_want, layer_type='fc'):
                        print('Layer fc %s init from file' % layer_name)
                        init = 'from_file'
                        init_file = getInitFileName(i, cfg, key_want=key_want, layer_type='fc')
                        init_layer_keys = getInitFileArgs(i, cfg, key_want=key_want, layer_type='fc')

                        if cache_filter == 1:
                            init = 'from_cached'
                            filter_cache_str_prefix = '%s_%i' % (var_name, i + var_offset)
                            weight_key = '%s/weight' % filter_cache_str_prefix
                            bias_key = '%s/bias' % filter_cache_str_prefix
                            if not weight_key in dict_cache_filter:
                                params = np.load(init_file)
                                dict_cache_filter[weight_key] = tf.constant(params[init_layer_keys['weight']],
                                                                            dtype=tf.float32)
                                dict_cache_filter[bias_key] = tf.constant(params[init_layer_keys['bias']],
                                                                          dtype=tf.float32)

                            init_layer_keys = {'weight': dict_cache_filter[weight_key],
                                               'bias': dict_cache_filter[bias_key]}

                    if getFcOutput(i, cfg, key_want=key_want):
                        if init=='trunc_norm':
                            init = init_type
                        new_encode_node = m.fc(getFcNumFilters(i, cfg, key_want=key_want),
                                               activation=None, dropout=None, bias=0, weight_decay=weight_decay,
                                               init=init, init_file=init_file, init_layer_keys=init_layer_keys)
                    else:
                        new_encode_node = m.fc(getFcNumFilters(i, cfg, key_want=key_want),
                                               dropout=getFcDropout(i, cfg, train, key_want=key_want), bias=.1,
                                               weight_decay=weight_decay,
                                               init=init, init_file=init_file, init_layer_keys=init_layer_keys)

                # do pool
                do_pool = getEncodeDoPool(i, cfg, key_want=key_want)
                if do_pool:
                    if 'encode' in layer_name:
                        all_out_dict[layer_name + '_bfpl'] = new_encode_node
                    pfs = getEncodePoolFilterSize(i, cfg, key_want=key_want)
                    ps = getEncodePoolStride(i, cfg, key_want=key_want)
                    pool_type = getEncodePoolType(i, cfg, key_want=key_want)
                    pool_padding = getPoolPadding(i, cfg, which_one=key_want)

                    if pool_type == 'max':
                        pfunc = 'maxpool'
                    elif pool_type == 'avg':
                        pfunc = 'avgpool'

                    new_encode_node = m.pool(pfs, ps, pfunc=pfunc, padding=pool_padding)
                    # print('Network %s %s pool %s with size %d stride %d' % (key_want, pfunc, layer_name, pfs, ps))

                if getWhetherSoftmax(i, cfg, key_want=key_want):
                    new_encode_node = m.softmax()

            # sm_add
            with tf.variable_scope(curr_reuse_name):
                # do up projection
                if getWhetherUpProj(i, cfg, key_want=key_want):
                    new_encode_node = m.upprojection(
                        up_settings=getUpProjSettings(i, cfg, key_want=key_want),
                        weight_decay=weight_decay, bias=0,
                        init=init_type, stddev=init_stddev,
                        train=train,
                        reuse_flag=reuse_flag,
                        batch_name=batch_name_new,
                        batch_reuse=reuse_batch_new,
                        tpu_flag=tpu_flag,
                    )

            if getWhetherBn(i, cfg, key_want=key_want):
                print("getWhetherBn appear:")
                # with tf.variable_scope('%s_bn%i%s' % (key_want, i, batch_name), reuse=reuse_batch):
                if tpu_flag==0:
                    with tf.variable_scope('%s_bn%i%s' % (var_name, i + var_offset, batch_name), reuse=reuse_batch):
                        new_encode_node = m.batchnorm_corr(train)
                else:
                    with tf.variable_scope('%s_bn%i%s' % (var_name, i + var_offset, batch_name), reuse=reuse_batch):
                        new_encode_node = m.tpu_batchnorm(train)


            if getWhetherFdb(i, cfg, key_want=key_want):
                from_layer = getFdbFrom(i, cfg, key_want=key_want)
                assert from_layer in all_out_dict, "Fdb nodes not built yet for network %s, layer %i!" % (key_want, i)
                with tf.variable_scope('%s_fdb%i' % (fdb_var_name, i + fdb_var_offset), reuse=fdb_reuse_flag):
                    new_encode_node = m.modulate(all_out_dict[from_layer], bias=0, init='trunc_norm',
                                                 stddev=init_stddev,
                                                 weight_decay=weight_decay)

            #print("{}_{}:".format(key_want, i), new_encode_node.shape)
            #print(new_encode_node.shape)
            #print("*********")
            all_out_dict[layer_name] = new_encode_node

    return m, all_out_dict


def build_datasetnet(
        inputs,
        cfg_initial,
        dataset_prefix,
        all_outputs=[],
        reuse_dict={},
        center_im=False,
        cfg_dataset={},
        no_prep=0,
        cache_filter=0,
        extra_feat=0,
        color_norm=0,
        add_batchname=None,
        tpu_tl_imagenet=None,
        tpu_task=None,
        color_dp_tl=False,
        rp_dp_tl=False,
        combine_tpu_flag=0,
        use_lasso=0,
        combine_col_rp=0,
        instance_task=False,
        instance_k=4096,
        instance_t=0.07,
        instance_m=0.5,
        instance_data_len=1281167,
        instance_lbl_pkl=None,
        instance_ret_out=False,
        inst_cate_sep=False,
        **kwargs):
    ret_params = None

    now_input_name = 'image_%s' % dataset_prefix

    if dataset_prefix == 'imagenet' and inst_cate_sep:
        instance_task = False

    if (cfg_dataset.get(dataset_prefix, 0) == 1 and tpu_task) \
            or (cfg_dataset.get(dataset_prefix, 0) == 1 \
                and now_input_name in inputs):
        # Preprocessing
        if tpu_task:
            if isinstance(inputs, dict):
                image_dataset = tf.cast(inputs[now_input_name], tf.float32)
            else:
                # If input is one tensor
                image_dataset = tf.cast(inputs, tf.float32)
            if color_norm==1:
                image_dataset = tf.div(image_dataset, tf.constant(255, dtype=tf.float32))
                image_dataset = tf.map_fn(ColorNormalize, image_dataset)
            elif color_norm==2:
                print("Scale to 0-1")
                image_dataset = tf.div(image_dataset, tf.constant(255, dtype=tf.float32))
        else:
            image_dataset = tf.cast(inputs[now_input_name], tf.float32)
        if no_prep==0 and tpu_task is None:
            image_dataset = tf.div(image_dataset, tf.constant(255, dtype=tf.float32))
            if center_im:
                image_dataset = tf.subtract(
                        image_dataset, 
                        tf.constant(0.5, dtype=tf.float32))
            if color_norm==1:
                image_dataset = tf.map_fn(ColorNormalize, image_dataset)
        
        if tpu_tl_imagenet is not None:
            dataset_prefix = tpu_tl_imagenet
        
        output_nodes = []
        all_out_dict_dataset = {}
        dict_cache_filter = {}

        curr_order = '%s_order' % dataset_prefix
        assert curr_order in cfg_initial
        network_order = cfg_initial.get(curr_order)

        if extra_feat == 1 and dataset_prefix in ['imagenet', 'place']:
            # If extra_feat is 1, then depth and normal branch will be added to imagenet and place dataset as outputs
            # Remember to skip them during calculating loss and calculating rep_loss!

            add_branch_list = ['depth', 'normal']

            # Check whether needed information is there
            for curr_add_branch in add_branch_list:
                assert '%s_order' % curr_add_branch in cfg_initial, 'Model cfg should include %s branch info!' % curr_add_branch

            # Work on adding the branches into network order
            for curr_add_branch in add_branch_list:
                add_network_order = cfg_initial.get('%s_order' % curr_add_branch)
                for add_network in add_network_order:
                    if add_network not in network_order:
                        network_order.append(add_network)

        for network_name in network_order:
            if cfg_initial[network_name].get("input", 0) != 0:
                input_now = None
                if cfg_initial[network_name]["input"] == 'different-prefix' or cfg_initial[network_name]["input"].find('prefix') != -1:
                    print("Right condition!")
                    cfg_initial[network_name]['input'] = '%s_prefix_2' % dataset_prefix

            else:
                input_now = image_dataset

                if color_dp_tl:
                    input_now = rgb_to_lab(input_now)
                    input_now = input_now[:,:,:,:1] - 50
                    print("*******color_dp_tl***********")
                    print(input_now)
                    if combine_col_rp==1:
                        input_now = tf.tile(input_now, [1, 1, 1, 3])
                if rp_dp_tl:
                    MEAN_RGB = [0.485, 0.456, 0.406] 
                    offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
                    input_now -= offset
                   
            var_name = getVarName(cfg_initial, key_want=network_name)
            reuse_name = '%s_reuse' % var_name
            reuse_curr = reuse_dict.get(reuse_name, None)

            fdb_var_name = getFdbVarName(cfg_initial, key_want=network_name)
            fdb_reuse_name = '_fdb_%s_reuse' % fdb_var_name
            fdb_reuse_curr = reuse_dict.get(fdb_reuse_name, None)

            curr_batch_name = '_%s' % dataset_prefix
            curr_batch_name = curr_batch_name + getBnVarName(cfg_initial, key_want=network_name)
            curr_batch_name_indict = '%s_bn_%s' % (var_name, curr_batch_name)
            reuse_batch = reuse_dict.get(curr_batch_name_indict, None)

            if dataset_prefix == 'nyuv2':
                curr_batch_name = '_%s' % 'pbrnet'
                reuse_batch = True

            if add_batchname is not None:
                if network_name.find('prefix') == -1:
                    curr_batch_name = add_batchname
            
            if combine_tpu_flag == 1 or tpu_task:
                reuse_curr = tf.AUTO_REUSE
                reuse_batch = tf.AUTO_REUSE

            m_curr, all_out_dict_dataset = build_partnet(
                input_now, cfg_initial=cfg_initial, key_want=network_name, reuse_flag=reuse_curr,
                fdb_reuse_flag=fdb_reuse_curr, reuse_batch=reuse_batch, batch_name=curr_batch_name,
                all_out_dict=all_out_dict_dataset, cache_filter=cache_filter,
                dict_cache_filter=dict_cache_filter,
                **kwargs)

            reuse_dict[reuse_name] = True
            reuse_dict[fdb_reuse_name] = True
            reuse_dict[curr_batch_name_indict] = True

            if use_lasso==1 and network_name=='encode':
                
                layer_group = []
                for i in range(10, 33):
                    layer_name = 'encode_%d' % i
                    layer_feature = all_out_dict_dataset[layer_name]
                    layer_feature = tf.expand_dims(layer_feature, 0)
                    #print("block3 layer shape:", layer_feature)
                    layer_group.append(layer_feature)
                layer_metric = tf.concat(layer_group, axis=0)
                
                lasso_metric_name = '%s_lasso' % 'colorization'
                with tf.variable_scope('encode_33', reuse=tf.AUTO_REUSE):
                    lasso_metric = tf.get_variable(shape=[23, 1, 1, 1, 1], dtype=tf.float32, name=lasso_metric_name)
                sum_metric = tf.reduce_sum(tf.square(lasso_metric))
                lasso_metric = lasso_metric / sum_metric
                
                # Use a trick to save the space
                '''
                layer_name = 'encode_10'
                layer_feature = all_out_dict_dataset[layer_name]
                weight_layer_feature = tf.multiply(lasso_metric[0], layer_feature)
                sum_layer_feature = weight_layer_feature 
                for i in range(11, 33):
                    layer_name = 'encode_%d' % i
                    layer_feature = all_out_dict_dataset[layer_name]
                    weight_layer_feature = tf.multiply(lasso_metric[i-10], layer_feature)
                    sum_layer_feature += weight_layer_feature
                '''
                m_curr.output = tf.multiply(lasso_metric, layer_metric)
                m_curr.output = tf.reduce_sum(m_curr.output, 0)
                #m_curr.output = sum_layer_feature
                print("lasso m_curr.output:", m_curr.output)
                all_out_dict_dataset['encode_33'] = m_curr.output

            
            as_output = cfg_initial.get(network_name).get('as_output', 0)
            if as_output == 1:
                print("output_size:{}".format(m_curr.output))
                full_size_resize = cfg_initial.get(network_name).get('resize', 0)
                if full_size_resize == 1:
                    m_curr.output = tf.image.resize_images(
                            m_curr.output,
                            (image_dataset.shape[1], image_dataset.shape[2]))
                output_nodes.append(m_curr.output)
                ret_params = m_curr.params
                outputs = m_curr.output

        all_outputs.extend(output_nodes)

        if instance_task or tpu_task=='instance_task':
            # Build memory bank
            ## Assuming the last output in all_outputs is the target
            curr_out = all_outputs.pop(-1)
            curr_out = tf.nn.l2_normalize(curr_out, axis=1) # [bs, out_dim]

            ## Actually define the memory bank (and label bank)
            var_scope_memory = dataset_prefix
            if dataset_prefix == 'imagenet_un':
                var_scope_memory = 'imagenet'
            with tf.variable_scope(var_scope_memory, reuse=tf.AUTO_REUSE):
                ### Set the variable and initial values
                batch_size, out_dim = curr_out.get_shape().as_list()
                model_seed = kwargs.get('seed', 0)

                #### Get initial memory bank
                if tpu_task is None:
                    mb_init = tf.random_uniform(
                            shape=(instance_data_len, out_dim),
                            seed=model_seed,
                            )
                else:
                    mb_init = np.random.uniform(
                            size=(instance_data_len, out_dim))
                    mb_init = mb_init.astype(np.float32)
                std_dev = 1. / np.sqrt(out_dim/3)
                mb_init = mb_init * (2*std_dev) - std_dev
                memory_bank = tf.get_variable(
                        'memory_bank', 
                        initializer=mb_init,
                        dtype=tf.float32,
                        trainable=False,
                        )

                #### Get initial all labels
                if instance_lbl_pkl is None:
                    label_init = tf.zeros_initializer
                    all_label_kwarg = {
                            'shape':(instance_data_len),
                            }
                else:
                    import cPickle
                    label_init = cPickle.load(open(instance_lbl_pkl, 'r'))
                    label_init = label_init.astype(np.int64)
                    all_label_kwarg = {}
                all_labels = tf.get_variable(
                        'all_labels',
                        initializer=label_init,
                        trainable=False,
                        dtype=tf.int64,
                        **all_label_kwarg
                        )

                is_training = kwargs.get('train', False)
                if is_training:
                    ### Randomly sample noise labels
                    index_name = 'index_%s' % dataset_prefix
                    assert index_name in inputs, "Input should include index!"
                    data_indx = inputs[index_name]
                    noise_indx = tf.random_uniform(
                            shape=(batch_size, instance_k),
                            minval=0,
                            maxval=instance_data_len,
                            dtype=tf.int64)
                    # data_memory: [bs, out_dim]
                    data_memory = tf.gather(memory_bank, data_indx, axis=0) 
                    # noise_memory [bs, k, out_dim]
                    noise_memory = tf.reshape(
                            tf.gather(memory_bank, noise_indx, axis=0),
                            [batch_size, instance_k, out_dim]
                            ) 
                    ### Compute the data distance and noise distance
                    curr_out_ext = tf.expand_dims(curr_out, axis=1)
                    data_dist = tf.reshape(
                            tf.matmul(
                                curr_out_ext, 
                                tf.expand_dims(data_memory, axis=2)), 
                            [batch_size]) # [bs]
                    noise_dist = tf.squeeze(
                            tf.matmul(
                                curr_out_ext, 
                                tf.transpose(noise_memory, [0,2,1])),
                            axis=1) # [bs, k]
                    data_dist = tf.exp(data_dist / instance_t)
                    noise_dist = tf.exp(noise_dist / instance_t)
                    instance_Z = tf.constant(
                            2876934.2 / 1281167 * instance_data_len, 
                            dtype=tf.float32)
                    data_dist /= instance_Z
                    noise_dist /= instance_Z
                    add_outputs = [data_dist, noise_dist]
                    ### Update the memory bank
                    new_data_memory = data_memory*instance_m \
                            + (1-instance_m)*curr_out
                    new_data_memory = tf.nn.l2_normalize(
                            new_data_memory, 
                            axis=1)
                    if not tpu_task:
                        add_outputs.extend(
                                [memory_bank, data_indx, new_data_memory])
                    else:
                        update_data_memory = new_data_memory - data_memory
                        scatter_memory = tf.scatter_nd(
                                tf.expand_dims(data_indx, axis=1),
                                update_data_memory,
                                shape=memory_bank.shape)
                        # On tpu, collecting all updates on each tpu core
                        scatter_memory = tf.contrib.tpu.cross_replica_sum(scatter_memory)
                        mb_update_op = tf.assign_add(
                                memory_bank, 
                                scatter_memory,
                                use_locking=False)
                        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mb_update_op)
                    ### Update the labels
                    if instance_lbl_pkl is None:
                        add_outputs.append(all_labels)
                else: # During validation
                    all_dist = tf.matmul(
                            curr_out, 
                            tf.transpose(memory_bank, [1, 0])) # [bs, data_len]
                    add_outputs = [all_dist, all_labels]
                    if instance_ret_out:
                        add_outputs.append(curr_out)

                all_outputs.extend(add_outputs)

    if tpu_task and dataset_prefix == 'rp':
        print("after resnet output:", m_curr.output)
        main_head = tf.reshape(
                m_curr.output, 
                [tf.div(m_curr.output.shape[0], 4), 4, 
                 m_curr.output.shape[1], m_curr.output.shape[2], 
                 m_curr.output.shape[3]])

        head_0_ = main_head[:,0,:,:,:]
        head_1_ = main_head[:,1,:,:,:]
        head_2_ = main_head[:,2,:,:,:]
        head_3_ = main_head[:,3,:,:,:]

        head_0 = tf.reshape(
                head_0_, 
                [head_0_.shape[0], 1, 1, \
                        head_0_.shape[1]*head_0_.shape[2]*head_0_.shape[3]])
        head_1 = tf.reshape(
                head_1_, 
                [head_1_.shape[0], 1, 1, \
                        head_1_.shape[1]*head_1_.shape[2]*head_1_.shape[3]])
        head_2 = tf.reshape(
                head_2_, 
                [head_2_.shape[0], 1, 1, \
                        head_2_.shape[1]*head_2_.shape[2]*head_2_.shape[3]])
        head_3 = tf.reshape(
                head_3_, 
                [head_3_.shape[0], 1, 1, \
                        head_3_.shape[1]*head_3_.shape[2]*head_3_.shape[3]])

        print("head_0 shape:", head_0)
        all_head = [] 
        for num_pair, head_pair in zip(
                [(0, 1), (1, 0), (1, 1), (1, -1), (1, 0), (0, 1)],
                [(head_0, head_1), (head_0, head_2), (head_0, head_3), \
                        (head_1, head_2), (head_1, head_3), (head_2, head_3)]):
            #print("head_pair_0:", head_pair[0])
            #print("head_pair_1:", head_pair[1])

            head_branch = tf.concat([head_pair[0], head_pair[1]], 3)

            all_head.append(head_branch)

            head_branch = tf.concat([head_pair[1], head_pair[0]], 3)

            all_head.append(head_branch)
        
        input_now  = tf.concat(all_head, 0)
        
        network_name = 'rp_category'

        print("concat:", input_now)

        var_name = getVarName(cfg_initial, key_want=network_name)
        reuse_name = '%s_reuse' % var_name
        reuse_curr = reuse_dict.get(reuse_name, None)

        fdb_var_name = getFdbVarName(cfg_initial, key_want=network_name)
        fdb_reuse_name = '_fdb_%s_reuse' % fdb_var_name
        fdb_reuse_curr = reuse_dict.get(fdb_reuse_name, None)

        curr_batch_name = '_%s' % dataset_prefix
        curr_batch_name = curr_batch_name \
                + getBnVarName(cfg_initial, key_want=network_name)
        curr_batch_name_indict = '%s_bn_%s' % (var_name, curr_batch_name)
        reuse_batch = reuse_dict.get(curr_batch_name_indict, None)
        if tpu_task:
            reuse_curr = tf.AUTO_REUSE
            reuse_batch = tf.AUTO_REUSE

        m_curr, all_out_dict_dataset = build_partnet(
            input_now, cfg_initial=cfg_initial, 
            key_want=network_name, reuse_flag=reuse_curr,
            fdb_reuse_flag=fdb_reuse_curr, 
            reuse_batch=reuse_batch, batch_name=curr_batch_name,
            all_out_dict=all_out_dict_dataset, cache_filter=cache_filter,
            dict_cache_filter=dict_cache_filter,
            **kwargs)

        #print("m_curr output:", m_curr.output)
        reuse_dict[reuse_name] = True
        reuse_dict[fdb_reuse_name] = True
        reuse_dict[curr_batch_name_indict] = True

        as_output = cfg_initial.get(network_name).get('as_output', 0)
 
        if as_output == 1:
            #print("output_size:{}".format(m_curr.output))
            full_size_resize = cfg_initial.get(network_name).get('resize', 0)
            if full_size_resize == 1:
                m_curr.output = tf.image.resize_images(
                        m_curr.output,
                        (image_dataset.shape[1], image_dataset.shape[2]))
            
            outputs = m_curr.output

        outputs = tf.reshape(outputs, [12, -1, 8])
        outputs = tf.transpose(outputs, [1, 0, 2])
        all_outputs = outputs
     
    if tpu_task:
        return all_outputs
    else:
        return all_outputs, reuse_dict, ret_params


def combine_tfutils_general(
        inputs, 
        mean_teacher=False, 
        ema_decay=0.9997,
        ema_zerodb=False,
        **kwargs):

    all_outputs = []
    reuse_dict = {}
    ret_params_final = None
    

    # dataset_prefix_list = ['scenenet', 'pbrnet', 'imagenet', 'coco', 'place']
    dataset_prefix_list = [
            'scenenet', 'pbrnet', 'imagenet', 'imagenet_un', 
            'coco', 'place', 'kinetics', 'rp', 'colorization']
    for dataset_prefix in dataset_prefix_list:
        print("dataset:{}".format(dataset_prefix))
        if not mean_teacher:
            all_outputs, reuse_dict, ret_params = build_datasetnet(
                    inputs, all_outputs=all_outputs, reuse_dict=reuse_dict,
                    dataset_prefix=dataset_prefix, **kwargs)
        else: 
            # Build the model in the special name scope, 
            # build teacher model for imagenet and imagenet_un
            with name_variable_scope("primary", "primary", reuse=tf.AUTO_REUSE) \
                    as (name_scope, var_scope):
                all_outputs, reuse_dict, ret_params = build_datasetnet(
                        inputs, all_outputs=all_outputs, reuse_dict=reuse_dict,
                        dataset_prefix=dataset_prefix, **kwargs)
            if dataset_prefix in ['imagenet', 'imagenet_un']:
                # Build the teacher model using ema_variable_scope
                with ema_variable_scope(
                        "ema", var_scope, decay=ema_decay, 
                        zero_debias=ema_zerodb, reuse=tf.AUTO_REUSE):
                    all_outputs, reuse_dict, ret_params = build_datasetnet(
                            inputs, all_outputs=all_outputs, 
                            reuse_dict=reuse_dict,
                            dataset_prefix=dataset_prefix, **kwargs)

        if not ret_params is None:
            ret_params_final = ret_params

    all_outputs, reuse_dict, _ = build_datasetnet(
            inputs, all_outputs=all_outputs, reuse_dict=reuse_dict,
            dataset_prefix='nyuv2', **kwargs)

    return all_outputs, ret_params_final

def tpu_combine_tfutils_general(inputs, 
        mean_teacher=False, 
        ema_decay=0.9997,
        ema_zerodb=False,
        **kwargs):
    reuse_dict = {}

    dataset_prefix_list = ['imagenet', 'imagenet_un']
    for dataset_prefix in dataset_prefix_list:
        print("dataset:{}".format(dataset_prefix))

        if not mean_teacher:
            outputs = build_datasetnet(
                    inputs, reuse_dict=reuse_dict,
                    dataset_prefix=dataset_prefix, **kwargs)
        else: 
            # Build the model in the special name scope, 
            # build teacher model for imagenet and imagenet_un
            reuse_now = tf.AUTO_REUSE

            with name_variable_scope("primary", "primary", reuse=reuse_now) \
                    as (name_scope, var_scope):
                outputs = build_datasetnet(
                        inputs, reuse_dict=reuse_dict,
                        dataset_prefix=dataset_prefix, **kwargs)
            if dataset_prefix in ['imagenet', 'imagenet_un']:
                # Build the teacher model using ema_variable_scope
                with ema_variable_scope(
                        "ema", var_scope, decay=ema_decay, 
                        zero_debias=ema_zerodb, reuse=reuse_now):
                    outputs = build_datasetnet(
                            inputs, reuse_dict=reuse_dict,
                            dataset_prefix=dataset_prefix, **kwargs)

    output_dict = {}
    for indx in xrange(len(outputs)):
        output_dict[indx] = outputs[indx]
    output_dict['logits'] = outputs[0]
    return output_dict

def split_input(inputs, n_gpus=1):
    if n_gpus == 1:
        return [inputs]

    temp_args = {v: tf.split(inputs[v], axis=0, num_or_size_splits=n_gpus) for v in inputs}
    list_of_args = [{now_arg: temp_args[now_arg][ind] for now_arg in temp_args} \
            for ind in xrange(n_gpus)]
    return list_of_args

'''
This is Siming's working park!
'''

def tpu_build_imagenet(
        inputs,
        cfg_initial,
        dataset_prefix,
        all_outputs=[],
        reuse_dict={},
        center_im=False,
        cfg_dataset={},
        no_prep=0,
        cache_filter=0,
        extra_feat=0,
        color_norm=0,
        add_batchname=None,
        tpu_task=None,
        use_lasso=0,
        combine_fewshot=0,
        #tpu_flag=0,
        **kwargs):
    ret_params = None
    if cfg_dataset.get(dataset_prefix, 0) == 1:

        images = tf.cast(inputs, tf.float32)

        output_nodes = []
        all_out_dict_dataset = {}
        dict_cache_filter = {}

        curr_order = '%s_order' % dataset_prefix
        assert curr_order in cfg_initial
        network_order = cfg_initial.get(curr_order)
        
        for network_name in network_order:
            if cfg_initial[network_name].get("input", 0) != 0:
                input_now = None
                if cfg_initial[network_name]["input"] == 'different-prefix' or cfg_initial[network_name]["input"].find('prefix') != -1:
                    print("Right condition!")
                    cfg_initial[network_name]['input'] = '%s_prefix_2' % dataset_prefix
            else:
                input_now = images

            var_name = getVarName(cfg_initial, key_want=network_name)
            reuse_name = '%s_reuse' % var_name
            reuse_curr = reuse_dict.get(reuse_name, None)

            fdb_var_name = getFdbVarName(cfg_initial, key_want=network_name)
            fdb_reuse_name = '_fdb_%s_reuse' % fdb_var_name
            fdb_reuse_curr = reuse_dict.get(fdb_reuse_name, None)

            curr_batch_name = '_%s' % dataset_prefix
            curr_batch_name = curr_batch_name + getBnVarName(cfg_initial, key_want=network_name)
            curr_batch_name_indict = '%s_bn_%s' % (var_name, curr_batch_name)
            reuse_batch = reuse_dict.get(curr_batch_name_indict, None)

            if dataset_prefix == 'nyuv2':
                curr_batch_name = '_%s' % 'pbrnet'
                reuse_batch = True

            if add_batchname is not None:
                curr_batch_name = add_batchname

            if tpu_task is not None:
                reuse_curr = tf.AUTO_REUSE
                reuse_batch = tf.AUTO_REUSE
            #print("*************reuse_curr***********")
            #print(reuse_curr)

            m_curr, all_out_dict_dataset = build_partnet(
                input_now, cfg_initial=cfg_initial, key_want=network_name, reuse_flag=reuse_curr,
                fdb_reuse_flag=fdb_reuse_curr, reuse_batch=reuse_batch, batch_name=curr_batch_name,
                all_out_dict=all_out_dict_dataset, cache_filter=cache_filter,
                dict_cache_filter=dict_cache_filter, combine_fewshot=combine_fewshot,
                **kwargs)

            reuse_dict[reuse_name] = True
            reuse_dict[fdb_reuse_name] = True
            reuse_dict[curr_batch_name_indict] = True

            if use_lasso==1 and network_name=='encode':
                
                layer_group = []
                for i in range(10, 33):
                    layer_name = 'encode_%d' % i
                    layer_feature = all_out_dict_dataset[layer_name]
                    layer_feature = tf.expand_dims(layer_feature, 0)
                    #print("block3 layer shape:", layer_feature)
                    layer_group.append(layer_feature)
                layer_metric = tf.concat(layer_group, axis=0)
                
                lasso_metric_name = '%s_lasso' % dataset_prefix
                with tf.variable_scope('encode_33', reuse=tf.AUTO_REUSE):
                    lasso_metric = tf.get_variable(shape=[23, 1, 1, 1, 1], dtype=tf.float32, name=lasso_metric_name)
                sum_metric = tf.reduce_sum(tf.square(lasso_metric))
                lasso_metric = lasso_metric / sum_metric
                
                # Use a trick to save the space
                '''
                layer_name = 'encode_10'
                layer_feature = all_out_dict_dataset[layer_name]
                weight_layer_feature = tf.multiply(lasso_metric[0], layer_feature)
                sum_layer_feature = weight_layer_feature 
                for i in range(11, 33):
                    layer_name = 'encode_%d' % i
                    layer_feature = all_out_dict_dataset[layer_name]
                    weight_layer_feature = tf.multiply(lasso_metric[i-10], layer_feature)
                    sum_layer_feature += weight_layer_feature
                '''
                m_curr.output = tf.multiply(lasso_metric, layer_metric)
                m_curr.output = tf.reduce_sum(m_curr.output, 0)
                #m_curr.output = sum_layer_feature
                print("lasso m_curr.output:", m_curr.output)
                all_out_dict_dataset['encode_33'] = m_curr.output

            as_output = cfg_initial.get(network_name).get('as_output', 0)
            if as_output == 1:
                print("output_size:{}".format(m_curr.output))
                full_size_resize = cfg_initial.get(network_name).get('resize', 0)
                if full_size_resize == 1:
                    m_curr.output = tf.image.resize_images(m_curr.output,
                                                           (image_dataset.shape[1], image_dataset.shape[2]))
                outputs = m_curr.output

    return outputs



def tpu_build_rp_imagenet(
        inputs,
        cfg_initial,
        dataset_prefix,
        all_outputs=[],
        reuse_dict={},
        center_im=False,
        cfg_dataset={},
        no_prep=0,
        cache_filter=0,
        extra_feat=0,
        color_norm=0,
        add_batchname=None,
        tpu_task=None,
        use_lasso=0,
        # tpu_flag=0,
        **kwargs):
    ret_params = None

    if cfg_dataset.get(dataset_prefix, 0) == 1:
        images = tf.cast(inputs, tf.float32)

        output_nodes = []
        all_out_dict_dataset = {}
        dict_cache_filter = {}

        curr_order = '%s_order' % dataset_prefix
        assert curr_order in cfg_initial
        network_order = cfg_initial.get(curr_order)

        for network_name in network_order:

            if cfg_initial[network_name].get("input", 0) != 0:
                input_now = None
            else:
                input_now = images
            var_name = getVarName(cfg_initial, key_want=network_name)
            reuse_name = '%s_reuse' % var_name
            reuse_curr = reuse_dict.get(reuse_name, None)

            fdb_var_name = getFdbVarName(cfg_initial, key_want=network_name)
            fdb_reuse_name = '_fdb_%s_reuse' % fdb_var_name
            fdb_reuse_curr = reuse_dict.get(fdb_reuse_name, None)

            curr_batch_name = '_%s' % dataset_prefix
            curr_batch_name = curr_batch_name + getBnVarName(cfg_initial, key_want=network_name)
            curr_batch_name_indict = '%s_bn_%s' % (var_name, curr_batch_name)
            reuse_batch = reuse_dict.get(curr_batch_name_indict, None)

            if dataset_prefix == 'nyuv2':
                curr_batch_name = '_%s' % 'pbrnet'
                reuse_batch = True

            if add_batchname is not None:
                curr_batch_name = add_batchname

            if tpu_task is not None:
                reuse_curr = tf.AUTO_REUSE
                reuse_batch = tf.AUTO_REUSE

            m_curr, all_out_dict_dataset = build_partnet(
                input_now, cfg_initial=cfg_initial, key_want=network_name, reuse_flag=reuse_curr,
                fdb_reuse_flag=fdb_reuse_curr, reuse_batch=reuse_batch, batch_name=curr_batch_name,
                all_out_dict=all_out_dict_dataset, cache_filter=cache_filter,
                dict_cache_filter=dict_cache_filter,
                **kwargs)

            reuse_dict[reuse_name] = True
            reuse_dict[fdb_reuse_name] = True
            reuse_dict[curr_batch_name_indict] = True
            if use_lasso==1 and network_name=='encode':
                
                layer_group = []
                for i in range(10, 33):
                    layer_name = 'encode_%d' % i
                    layer_feature = all_out_dict_dataset[layer_name]
                    layer_feature = tf.expand_dims(layer_feature, 0)
                    #print("block3 layer shape:", layer_feature)
                    layer_group.append(layer_feature)
                layer_metric = tf.concat(layer_group, axis=0)
                
                lasso_metric_name = '%s_lasso' % dataset_prefix
                with tf.variable_scope('encode_33', reuse=tf.AUTO_REUSE):
                    lasso_metric = tf.get_variable(shape=[23, 1, 1, 1, 1], dtype=tf.float32, name=lasso_metric_name)
                sum_metric = tf.reduce_sum(tf.square(lasso_metric))
                lasso_metric = lasso_metric / sum_metric
                
                # Use a trick to save the space
                '''
                layer_name = 'encode_10'
                layer_feature = all_out_dict_dataset[layer_name]
                weight_layer_feature = tf.multiply(lasso_metric[0], layer_feature)
                sum_layer_feature = weight_layer_feature 
                for i in range(11, 33):
                    layer_name = 'encode_%d' % i
                    layer_feature = all_out_dict_dataset[layer_name]
                    weight_layer_feature = tf.multiply(lasso_metric[i-10], layer_feature)
                    sum_layer_feature += weight_layer_feature
                '''
                m_curr.output = tf.multiply(lasso_metric, layer_metric)
                m_curr.output = tf.reduce_sum(m_curr.output, 0)
                #m_curr.output = sum_layer_feature
                print("lasso m_curr.output:", m_curr.output)
                all_out_dict_dataset['encode_33'] = m_curr.output

        
        print("after resnet output:", m_curr.output)
        main_head = tf.reshape(m_curr.output, [tf.div(m_curr.output.shape[0], 4), 4, m_curr.output.shape[1], m_curr.output.shape[2], m_curr.output.shape[3]])

        head_0_ = main_head[:,0,:,:,:]
        head_1_ = main_head[:,1,:,:,:]
        head_2_ = main_head[:,2,:,:,:]
        head_3_ = main_head[:,3,:,:,:]

        head_0 = tf.reshape(head_0_, [head_0_.shape[0], 1, 1, head_0_.shape[1]*head_0_.shape[2]*head_0_.shape[3]])
        head_1 = tf.reshape(head_1_, [head_1_.shape[0], 1, 1, head_1_.shape[1]*head_1_.shape[2]*head_1_.shape[3]])
        head_2 = tf.reshape(head_2_, [head_2_.shape[0], 1, 1, head_2_.shape[1]*head_2_.shape[2]*head_2_.shape[3]])
        head_3 = tf.reshape(head_3_, [head_3_.shape[0], 1, 1, head_3_.shape[1]*head_3_.shape[2]*head_3_.shape[3]])

        print("head_0 shape:", head_0)
        all_head = [] 
        for num_pair, head_pair in zip([(0, 1), (1, 0), (1, 1), (1, -1), (1, 0), (0, 1)],[(head_0, head_1), (head_0, head_2), (head_0, head_3), (head_1, head_2), (head_1, head_3), (head_2, head_3)]):
            #print("head_pair_0:", head_pair[0])
            #print("head_pair_1:", head_pair[1])

            head_branch = tf.concat([head_pair[0], head_pair[1]], 3)

            all_head.append(head_branch)

            head_branch = tf.concat([head_pair[1], head_pair[0]], 3)

            all_head.append(head_branch)
        
        input_now  = tf.concat(all_head, 0)
        
        network_name = 'rp_category'

        print("concat:", input_now)

        var_name = getVarName(cfg_initial, key_want=network_name)
        reuse_name = '%s_reuse' % var_name
        reuse_curr = reuse_dict.get(reuse_name, None)

        fdb_var_name = getFdbVarName(cfg_initial, key_want=network_name)
        fdb_reuse_name = '_fdb_%s_reuse' % fdb_var_name
        fdb_reuse_curr = reuse_dict.get(fdb_reuse_name, None)

        curr_batch_name = '_%s' % dataset_prefix
        curr_batch_name = curr_batch_name + getBnVarName(cfg_initial, key_want=network_name)
        curr_batch_name_indict = '%s_bn_%s' % (var_name, curr_batch_name)
        reuse_batch = reuse_dict.get(curr_batch_name_indict, None)

        if tpu_task is not None:
            reuse_curr = tf.AUTO_REUSE
            reuse_batch = tf.AUTO_REUSE

        m_curr, all_out_dict_dataset = build_partnet(
            input_now, cfg_initial=cfg_initial, key_want=network_name, reuse_flag=reuse_curr,
            fdb_reuse_flag=fdb_reuse_curr, reuse_batch=reuse_batch, batch_name=curr_batch_name,
            all_out_dict=all_out_dict_dataset, cache_filter=cache_filter,
            dict_cache_filter=dict_cache_filter,
            **kwargs)

        #print("m_curr output:", m_curr.output)
        reuse_dict[reuse_name] = True
        reuse_dict[fdb_reuse_name] = True
        reuse_dict[curr_batch_name_indict] = True

        as_output = cfg_initial.get(network_name).get('as_output', 0)
 
        if as_output == 1:
            #print("output_size:{}".format(m_curr.output))
            full_size_resize = cfg_initial.get(network_name).get('resize', 0)
            if full_size_resize == 1:
                m_curr.output = tf.image.resize_images(m_curr.output,
                                                           (image_dataset.shape[1], image_dataset.shape[2]))
            
            outputs = m_curr.output

        outputs = tf.reshape(outputs, [12, -1, 8])
        outputs = tf.transpose(outputs, [1, 0, 2])

    return outputs

def tpu_combine_tfutils_imagenet(inputs, **kwargs):
    print("inputs of model function:", inputs.shape)
    #print("Building the depth network!")
    logits = tpu_build_imagenet(inputs, dataset_prefix='imagenet', **kwargs)
    print("output of model function:", logits)

    return logits

def tpu_combine_tfutils_imagenet_rp(inputs, **kwargs):
    print("inputs of model function:", inputs.shape)
    #print("Building the depth network!")
    logits = tpu_build_imagenet(inputs, dataset_prefix='rp', **kwargs)
    print("output of model function:", logits)

    return logits

def tpu_combine_tfutils_rp(inputs, **kwargs):
    inputs = tf.reshape(inputs, [inputs.shape[0]*inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4]])
    print("inputs of model function:", inputs.shape)
    #print("Building the depth network!")

    logits = tpu_build_rp_imagenet(inputs, dataset_prefix='rp', **kwargs)
    print("output of model function:", logits)

    return logits

def tpu_combine_tfutils_col(inputs, **kwargs):

    print("inputs of model function:", inputs.shape)
    #print("Building the depth network!")

    logits = tpu_build_imagenet(inputs, dataset_prefix='colorization', **kwargs)
    print("output of model function:", logits)

    return logits

def gpu_combine_tfutils_col(inputs, down_sample=8, col_knn=False, **kwargs):


    #print("Building the depth network!")
    inputs = inputs['image_imagenet']
    print("inputs of model function:", inputs)
    l_image, ab_image, Q_label = col_preprocess_for_gpu(inputs, down_sample=down_sample, col_knn=col_knn)  
    logits_dict = {}

    logits = tpu_build_col_imagenet(l_image, dataset_prefix='colorization', **kwargs)
    logits_dict['l'] = l_image
    logits_dict['logits'] = logits
    logits_dict['Q'] = Q_label
    logits_dict['ab_image'] = ab_image
    #logits['raw_image'] = inputs
    print("output of model function:", logits_dict)

    return logits_dict, {}

def gpu_combine_tfutils_col_tl(inputs, down_sample=8, col_knn=False, **kwargs):


    #print("Building the depth network!")
    inputs_ = inputs['image_imagenet']
    print("inputs of model function:", inputs)
    l_image, ab_image, Q_label = col_preprocess_for_gpu(inputs_, down_sample=down_sample, col_knn=col_knn)  
    logits_dict = {}

    logits = tpu_build_col_imagenet(l_image, dataset_prefix='imagenet', **kwargs)
    logits_dict['l'] = l_image
    logits_dict['logits'] = logits
    logits_dict['Q'] = inputs['label_imagenet']
    logits_dict['ab_image'] = ab_image
    #logits['raw_image'] = inputs
    print("output of model function:", logits_dict)

    return logits_dict, {}

def tpu_combine_tfutils_depth(inputs, **kwargs):
    print("inputs of model function:", inputs.shape)

    #print("Building the depth network!")
    logits = tpu_build_imagenet(inputs, dataset_prefix='pbrnet', **kwargs)
    print("output of model function:", logits)

    return logits

def tpu_combine_tfutils_depth_imn(inputs, **kwargs):
    print("inputs of model function:", inputs)

    #print("Building the depth network!")
    mlt_logits = tpu_build_imagenet(inputs['mlt'], dataset_prefix='pbrnet', **kwargs)
    imn_logits = tpu_build_imagenet(inputs['imn'], dataset_prefix='imagenet', **kwargs)
    logits = {}
    logits['mlt'] = mlt_logits
    logits['imn'] = imn_logits
    print("output of model function:", logits)
    depth_loss = tf.nn.l2_loss(logits['mlt'] - inputs['depth']) / np.prod(inputs['depth'].get_shape().as_list())
    one_hot_labels = tf.one_hot(inputs['imn_label'], 1000)
    imnet_loss = tf.losses.softmax_cross_entropy(logits=logits['imn'], onehot_labels=one_hot_labels) 
    loss = depth_loss + imnet_loss
    #print(loss)
    print(loss)
    loss = tf.expand_dims(loss, 0)
    loss = tf.tile(loss, [8])
    print(loss)
    loss = tf.reshape(loss, [8, 1])
    loss = tf.tile(loss, [1, 1000])
    print(loss)
    return loss

def tpu_combine_tfutils_rp_imn(inputs, **kwargs):
    
    '''RP Process'''
    rp_inputs = tf.reshape(inputs['rp'], [inputs['rp'].shape[0]*inputs['rp'].shape[1], inputs['rp'].shape[2], inputs['rp'].shape[3], inputs['rp'].shape[4]])
    rp_logits = tpu_build_rp_imagenet(rp_inputs, dataset_prefix='rp', **kwargs)
    print("output of model function:", rp_logits)
    
    pair = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1),(1,0),(-1,0),(0,1),(0,-1)]
    e_bs = tf.cast(rp_logits.shape[0], tf.int32)
    all_labels = []
    for i in range(0, 12):
        one_hot_labels = tf.one_hot(pos2lbl(pair[i]), 8)
        one_hot_labels = tf.reshape(one_hot_labels, [1, 1, 8])
        one_hot_labels = tf.tile(one_hot_labels, [e_bs, 1, 1])
        all_labels.append(one_hot_labels)

    all_labels = tf.concat(all_labels, axis=1)
    print("training loss labels:", all_labels)

    rp_loss = tf.losses.softmax_cross_entropy(logits=rp_logits, onehot_labels=all_labels)

    '''Imagenet Process'''
    imn_logits = tpu_build_imagenet(inputs['imn'], dataset_prefix='imagenet', **kwargs) 
    imn_one_hot_labels = tf.one_hot(inputs['labels'], 1000)
    imnet_loss = tf.losses.softmax_cross_entropy(logits=imn_logits, onehot_labels=imn_one_hot_labels)
    loss = rp_loss + imnet_loss
    loss = tf.expand_dims(loss, 0)
    loss = tf.tile(loss, [8])
    loss = tf.reshape(loss, [8, 1])
    loss = tf.tile(loss, [1, 1000])
    return loss

def tpu_combine_tfutils_rp_col(inputs, **kwargs):
    
    '''RP Process'''
    rp_inputs = tf.reshape(inputs['rp'], [inputs['rp'].shape[0]*inputs['rp'].shape[1], inputs['rp'].shape[2], inputs['rp'].shape[3], inputs['rp'].shape[4]])
    rp_logits = tpu_build_rp_imagenet(rp_inputs, dataset_prefix='rp', **kwargs)
    print("output of model function:", rp_logits)
    
    pair = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1),(1,0),(-1,0),(0,1),(0,-1)]
    e_bs = tf.cast(rp_logits.shape[0], tf.int32)
    all_labels = []
    for i in range(0, 12):
        one_hot_labels = tf.one_hot(pos2lbl(pair[i]), 8)
        one_hot_labels = tf.reshape(one_hot_labels, [1, 1, 8])
        one_hot_labels = tf.tile(one_hot_labels, [e_bs, 1, 1])
        all_labels.append(one_hot_labels)

    all_labels = tf.concat(all_labels, axis=1)
    print("training loss labels:", all_labels)

    rp_loss = tf.losses.softmax_cross_entropy(logits=rp_logits, onehot_labels=all_labels)

    '''Color Process'''
    color_logits = tpu_build_imagenet(inputs['col'], dataset_prefix='colorization', **kwargs) 
    flatten_color_logits = tf.reshape(color_logits, [-1, 313])
    flatten_labels = tf.reshape(inputs['col_labels'], [-1, 313])

    color_loss = tf.losses.softmax_cross_entropy(logits=flatten_color_logits, onehot_labels=flatten_labels)
    loss = rp_loss + color_loss
    loss = tf.expand_dims(loss, 0)
    loss = tf.tile(loss, [8])
    loss = tf.reshape(loss, [8, 1])
    loss = tf.tile(loss, [1, 1000])
    return loss

def tpu_combine_tfutils_rdc(inputs, **kwargs):
    
    '''RP Process'''
    rp_inputs = tf.reshape(inputs['rp_image'], [inputs['rp_image'].shape[0]*inputs['rp_image'].shape[1], inputs['rp_image'].shape[2], inputs['rp_image'].shape[3], inputs['rp_image'].shape[4]])
    rp_logits = tpu_build_rp_imagenet(rp_inputs, dataset_prefix='rp', **kwargs)
    print("output of model function:", rp_logits)

    pair = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1),(1,0),(-1,0),(0,1),(0,-1)]
    e_bs = tf.cast(rp_logits.shape[0], tf.int32)
    all_labels = []
    for i in range(0, 12):
        one_hot_labels = tf.one_hot(pos2lbl(pair[i]), 8)
        one_hot_labels = tf.reshape(one_hot_labels, [1, 1, 8])
        one_hot_labels = tf.tile(one_hot_labels, [e_bs, 1, 1])
        all_labels.append(one_hot_labels)

    all_labels = tf.concat(all_labels, axis=1)
    print("training loss labels:", all_labels)

    rp_loss = tf.losses.softmax_cross_entropy(logits=rp_logits, onehot_labels=all_labels)
    print("RP Process Done!")

    '''Color Process'''
    color_logits = tpu_build_imagenet(inputs['color_image'], dataset_prefix='colorization', **kwargs) 
    flatten_color_logits = tf.reshape(color_logits, [-1, 313])
    flatten_labels = tf.reshape(inputs['color_label'], [-1, 313])

    color_loss = tf.losses.softmax_cross_entropy(logits=flatten_color_logits, onehot_labels=flatten_labels)
    
    print("Color Process Done!")

    '''Depth Process'''
    depth_logits = tpu_build_imagenet(inputs['depth_image'], dataset_prefix='pbrnet', **kwargs)
    depth_loss = tf.nn.l2_loss(depth_logits - inputs['depth_label']) / np.prod(inputs['depth_label'].get_shape().as_list())
 
    loss = rp_loss + color_loss + depth_loss
    loss = tf.expand_dims(loss, 0)
    loss = tf.tile(loss, [8])
    loss = tf.reshape(loss, [8, 1])
    loss = tf.tile(loss, [1, 1000])

    print("Depth Process Done!")

    return loss

def tpu_combine_tfutils_rci(inputs, **kwargs):
    
    '''RP Process'''
    rp_inputs = tf.reshape(inputs['rp'], [inputs['rp'].shape[0]*inputs['rp'].shape[1], inputs['rp'].shape[2], inputs['rp'].shape[3], inputs['rp'].shape[4]])
    rp_logits = tpu_build_rp_imagenet(rp_inputs, dataset_prefix='rp', **kwargs)
    print("output of model function:", rp_logits)
    
    pair = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1),(1,0),(-1,0),(0,1),(0,-1)]
    e_bs = tf.cast(rp_logits.shape[0], tf.int32)
    all_labels = []
    for i in range(0, 12):
        one_hot_labels = tf.one_hot(pos2lbl(pair[i]), 8)
        one_hot_labels = tf.reshape(one_hot_labels, [1, 1, 8])
        one_hot_labels = tf.tile(one_hot_labels, [e_bs, 1, 1])
        all_labels.append(one_hot_labels)

    all_labels = tf.concat(all_labels, axis=1)
    print("training loss labels:", all_labels)

    rp_loss = tf.losses.softmax_cross_entropy(logits=rp_logits, onehot_labels=all_labels)

    '''Color Process'''
    color_logits = tpu_build_imagenet(inputs['col'], dataset_prefix='colorization', **kwargs) 
    flatten_color_logits = tf.reshape(color_logits, [-1, 313])
    flatten_labels = tf.reshape(inputs['col_labels'], [-1, 313])

    color_loss = tf.losses.softmax_cross_entropy(logits=flatten_color_logits, onehot_labels=flatten_labels)

    '''Imn Process'''
    imn_logits = tpu_build_imagenet(inputs['imn_images'], dataset_prefix='imagenet', **kwargs)
    imn_one_hot_labels = tf.one_hot(inputs['imn_labels'], 1000)
    imnet_loss = tf.losses.softmax_cross_entropy(logits=imn_logits, onehot_labels=imn_one_hot_labels)
    
    loss = rp_loss + color_loss + imnet_loss
    loss = tf.expand_dims(loss, 0)
    loss = tf.tile(loss, [8])
    loss = tf.reshape(loss, [8, 1])
    loss = tf.tile(loss, [1, 1000])
    return loss
