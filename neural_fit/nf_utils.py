from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import copy
import pdb

from sklearn import linear_model # for model_type = 0,1
from sklearn import cross_decomposition # for model_type = 2
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def expand_nodes(nodes):
    # This function will expand prefix_[a:b:c] to prefix_a,prefix_a+b,prefix_a+2b,...,prefix_a+c-b
    sp_res = nodes.split(',')
    for curr_i, each_sp in enumerate(sp_res):
        if '[' not in each_sp:
            continue
        sta_i = each_sp.find('[')
        end_i = each_sp.find(']')
        num_str = each_sp[sta_i+1 : end_i]
        sp_num = num_str.split(':')
        assert len(sp_num)==3, "Format of %s not correct!" % num_str
        sp_num = [int(tmp_s) for tmp_s in sp_num]
        a,b,c = sp_num
        prefix = each_sp[:sta_i]
        suffix = each_sp[end_i+1:]
        new_sp = ''
        for tmp_i in range(a,c,b):
            new_sp += "%s%i%s," % (prefix, tmp_i, suffix)
        new_sp = new_sp[:-1]

        sp_res[curr_i] = new_sp

    ret_str = ''
    for each_sp in sp_res:
        ret_str += '%s,' % each_sp
    ret_str = ret_str[:-1]
    return ret_str

def get_data_path(
        image_prefix='/mnt/fs0/datasets/neural_data/img_split/',
        neuron_resp_prefix='/mnt/fs1/Dataset/neural_resp/',
        ):
    DATA_PATH = {}

    for curr_split in range(5):
        if curr_split==0:
            curr_dir = 'V4IT'
        else:
            curr_dir = 'V4IT_split_%i' % (curr_split+1)

        DATA_PATH['split_%i/images' % curr_split] = os.path.join(image_prefix, curr_dir, 'tf_records', 'images')
        DATA_PATH['split_%i/IT' % curr_split] = os.path.join(neuron_resp_prefix, curr_dir, 'IT_ave')
        DATA_PATH['split_%i/V4' % curr_split] = os.path.join(neuron_resp_prefix, curr_dir, 'V4_ave')

    return DATA_PATH


def get_deepmind_data_path(
        image_prefix='/mnt/fs1/siming/Dataset/deepmind/',
        neuron_resp_prefix='/mnt/fs1/Dataset/neural_resp/',
        ):
    DATA_PATH = {}
    curr_split = 0
    curr_dir = 'V4IT'
    block_list = ['block2_1', 'block2_2', 'block2_3', 'block3_1', 'block3_2', 'block3_3', 'block2_4', 'block3_4']
    for block in block_list:
        DATA_PATH['split_%i/%s' % (curr_split, block)] = os.path.join(image_prefix, block)
    DATA_PATH['split_%i/IT' % curr_split] = os.path.join(neuron_resp_prefix, curr_dir, 'IT_ave')
    DATA_PATH['split_%i/V4' % curr_split] = os.path.join(neuron_resp_prefix, curr_dir, 'V4_ave')

    return DATA_PATH

def get_data_path_10ms(
        image_prefix='/mnt/fs0/datasets/neural_data/10ms/img_split/',
        neuron_resp_prefix='/mnt/fs0/datasets/neural_data/10ms/img_split/',
        ):
    DATA_PATH = {}

    for curr_split in range(5):

        curr_dir = 'V4IT_split%i' % (curr_split+1)

        DATA_PATH['split_%i/images' % curr_split] = os.path.join(image_prefix, curr_dir, 'tf_records', 'images')
        DATA_PATH['split_%i/labels' % curr_split] = os.path.join(neuron_resp_prefix, curr_dir, 'tf_records', 'labels')

    return DATA_PATH

def compute_corr(x, y):
    """
    Computes correlation of input vectors, using tf operations
    """
    x = tf.cast(tf.reshape(x, shape=(-1,)), tf.float32) # reshape and cast to ensure compatibility
    y = tf.cast(tf.reshape(y, shape=(-1,)), tf.float32)
    real_mask_x = tf.math.logical_not(tf.math.is_nan(x))
    real_mask_y = tf.math.logical_not(tf.math.is_nan(y))
    real_mask = tf.math.logical_and(real_mask_x, real_mask_y)
    x = tf.boolean_mask(x, real_mask)
    y = tf.boolean_mask(y, real_mask)
    #assert x.get_shape() == y.get_shape(), (x.get_shape(), y.get_shape())
    n = tf.cast(tf.shape(x)[0], tf.float32)
    x_sum = tf.reduce_sum(x)
    y_sum = tf.reduce_sum(y)
    xy_sum = tf.reduce_sum(tf.multiply(x, y))
    x2_sum = tf.reduce_sum(tf.pow(x, 2))
    y2_sum = tf.reduce_sum(tf.pow(y, 2))
    numerator = tf.scalar_mul(n, xy_sum) - tf.scalar_mul(x_sum, y_sum)
    denominator = tf.sqrt(tf.scalar_mul(tf.scalar_mul(n, x2_sum) - tf.pow(x_sum, 2),
				       tf.scalar_mul(n, y2_sum) - tf.pow(y_sum, 2))) + tf.constant(1e-4)
    corr = tf.truediv(numerator, denominator)

    return corr

def l2_and_corr(x, y):
    return tf.nn.l2_loss(x - y)/np.prod(y.get_shape().as_list()) - compute_corr(x, y) # we subtract the correlation to minimize this loss

def only_l2(x, y):
    return tf.nn.l2_loss(x - y)/np.prod(y.get_shape().as_list())

def loss_withcfg_10ms(output, labels, **kwargs):
    v4_nodes = kwargs.get('v4_nodes')
    it_nodes = kwargs.get('it_nodes')

    f10ms_time = kwargs.get('f10ms_time')
    time_list = [int(x) for x in f10ms_time.split(',')]

    all_losses = []

    if v4_nodes is not None:
        v4_node_list = v4_nodes.split(',')
        for each_node in v4_node_list:
            for which_time in time_list: 
                curr_v4_resps = labels[:, :88, which_time]
                all_losses.append(l2_and_corr(curr_v4_resps, output['v4_%i/%s' % (which_time, each_node)]))

    if it_nodes is not None:
        it_node_list = it_nodes.split(',')
        for each_node in it_node_list:
            for which_time in time_list: 
                curr_it_resps = labels[:, 88:, which_time]
                all_losses.append(l2_and_corr(curr_it_resps, output['it_%i/%s' % (which_time, each_node)]))

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if len(reg_losses)!=0:
        reg_losses = tf.add_n(reg_losses)
        all_losses.append(tf.cast(reg_losses, tf.float32))

    return tf.add_n(all_losses)

def loss_withcfg(output, *args, **kwargs):
    v4_nodes = kwargs.get('v4_nodes')
    it_nodes = kwargs.get('it_nodes')

    if len(args) == 2:
        v4_resps, it_resps = args
    else:
        v4_resps = args[0]
        it_resps = None

    all_losses = []

    v4_output = []
    if v4_nodes is not None:
        v4_node_list = v4_nodes.split(',')
        for each_node in v4_node_list:
            v4_output.append(output['v4/%s' % each_node])
            all_losses.append(l2_and_corr(v4_resps, output['v4/%s' % each_node]))

    it_output = []
    if it_nodes is not None:
        it_node_list = it_nodes.split(',')
        for each_node in it_node_list:
            it_output.append(output['it/%s' % each_node])
            all_losses.append(l2_and_corr(it_resps, output['it/%s' % each_node]))

    total_loss = tf.add_n(all_losses)
    #print_op = tf.print(
    #        'Total loss: ', total_loss,
    #        'Avg resp: ', tf.reduce_mean(it_resps),
    #        'Std resp: ', tf.math.reduce_std(it_resps),
    #        'Avg pred: ', tf.reduce_mean(it_output),
    #        'Std pred: ', tf.math.reduce_std(it_output),
    #        'L2 loss: ', only_l2(it_resps, it_output[0]))
    #print_op = tf.print(
    #        'Total loss: ', total_loss,
    #        'Avg resp: ', tf.reduce_mean(v4_resps),
    #        'Std resp: ', tf.math.reduce_std(v4_resps),
    #        'Avg pred: ', tf.reduce_mean(v4_output),
    #        'Std pred: ', tf.math.reduce_std(v4_output),
    #        'L2 loss: ', only_l2(v4_resps, v4_output[0]))
    print_op = tf.no_op()

    with tf.control_dependencies([print_op]):
        total_loss = tf.identity(total_loss)
    return total_loss

def loss_gather(
        inputs, 
        output, 
        **kwargs):
    all_losses = copy.copy(output)
    if 'V4_ave' in inputs:
        all_losses['V4_ave'] = inputs['V4_ave']
        all_losses['IT_ave'] = inputs['IT_ave']

    return all_losses

def loss_keep(
        inputs, 
        output, 
        **kwargs):
    return output

def loss_rep_withcfg_10ms(
        inputs, 
        output, 
        target, 
        v4_nodes=None,
        it_nodes=None,
        f10ms_time=None,
        **kwargs):

    all_resps = inputs['labels']

    all_losses = {}

    all_losses['corr/number'] = tf.shape(all_resps)[0]
    time_list = [int(x) for x in f10ms_time.split(',')]

    if v4_nodes is not None:
        v4_node_list = v4_nodes.split(',')

        for which_time in time_list:
            v4_resps = all_resps[:, :88, which_time]
            all_losses['corr/v4_%i/sum' % which_time] = tf.reduce_sum(v4_resps, axis = 0)
            all_losses['corr/v4_%i/square_sum' % which_time] = tf.reduce_sum(tf.pow(v4_resps, 2), axis = 0)

            for each_node in v4_node_list:
                curr_output = output['v4_%i/%s' % (which_time, each_node)]
                all_losses['l2_loss/v4_%i/%s' % (which_time, each_node)] = l2_and_corr(v4_resps, curr_output)
                all_losses['corr/v4_%i/%s/sum' % (which_time, each_node)] = tf.reduce_sum(curr_output, axis = 0)
                all_losses['corr/v4_%i/%s/square_sum' % (which_time, each_node)] = tf.reduce_sum(
                        tf.pow(curr_output, 2), axis = 0)
                all_losses['corr/v4_%i/%s/mult_sum' % (which_time, each_node)] = tf.reduce_sum(
                        tf.multiply(v4_resps, curr_output), axis = 0)

    if it_nodes is not None:
        it_node_list = it_nodes.split(',')
        
        for which_time in time_list:
            it_resps = all_resps[:, 88:, which_time]
            all_losses['corr/it_%i/sum' % which_time] = tf.reduce_sum(it_resps, axis = 0)
            all_losses['corr/it_%i/square_sum' % which_time] = tf.reduce_sum(tf.pow(it_resps, 2), axis = 0)

            for each_node in it_node_list:
                curr_output = output['it_%i/%s' % (which_time, each_node)]
                all_losses['l2_loss/it_%i/%s' % (which_time, each_node)] = l2_and_corr(it_resps, curr_output)
                all_losses['corr/it_%i/%s/sum' % (which_time, each_node)] = tf.reduce_sum(curr_output, axis = 0)
                all_losses['corr/it_%i/%s/square_sum' % (which_time, each_node)] = tf.reduce_sum(
                        tf.pow(curr_output, 2), axis = 0)
                all_losses['corr/it_%i/%s/mult_sum' % (which_time, each_node)] = tf.reduce_sum(
                        tf.multiply(it_resps, curr_output), axis = 0)

    return all_losses

def loss_rep_withcfg(
        inputs, 
        output, 
        target, 
        v4_nodes=None,
        it_nodes=None,
        **kwargs):

    if 'V4_ave' in inputs:
        v4_resps = inputs['V4_ave']
        it_resps = inputs['IT_ave']
    else:
        # V1V2 data
        v4_resps = inputs['V1_ave']
        it_resps = inputs.get('V2_ave', None)

    all_losses = {}

    all_losses['corr/number'] = tf.shape(v4_resps)[0]

    if v4_nodes is not None:
        v4_node_list = v4_nodes.split(',')
        all_losses['corr/v4/sum'] = tf.reduce_sum(v4_resps, axis = 0)
        all_losses['corr/v4/square_sum'] = tf.reduce_sum(tf.pow(v4_resps, 2), axis = 0)

        for each_node in v4_node_list:
            curr_output = output['v4/%s' % each_node]
            all_losses['l2_loss/v4/%s' % each_node] = l2_and_corr(v4_resps, curr_output)
            all_losses['corr/v4/%s/sum' % each_node] = tf.reduce_sum(curr_output, axis = 0)
            all_losses['corr/v4/%s/square_sum' % each_node] = tf.reduce_sum(tf.pow(curr_output, 2), axis = 0)
            all_losses['corr/v4/%s/mult_sum' % each_node] = tf.reduce_sum(tf.multiply(v4_resps, curr_output), axis = 0)

    if it_nodes is not None:
        it_node_list = it_nodes.split(',')
        all_losses['corr/it/sum'] = tf.reduce_sum(it_resps, axis = 0)
        all_losses['corr/it/square_sum'] = tf.reduce_sum(tf.pow(it_resps, 2), axis = 0)

        for each_node in it_node_list:
            curr_output = output['it/%s' % each_node]
            all_losses['l2_loss/it/%s' % each_node] = l2_and_corr(it_resps, curr_output)
            all_losses['corr/it/%s/sum' % each_node] = tf.reduce_sum(curr_output, axis = 0)
            all_losses['corr/it/%s/square_sum' % each_node] = tf.reduce_sum(tf.pow(curr_output, 2), axis = 0)
            all_losses['corr/it/%s/mult_sum' % each_node] = tf.reduce_sum(tf.multiply(it_resps, curr_output), axis = 0)

    return all_losses

def postprocess_config(cfg):
    cfg = copy.deepcopy(cfg)
    for k in cfg.get("network_list", ["decode", "encode"]):
        if k in cfg:
            ks = cfg[k].keys()
            for _k in ks:
                if _k.isdigit():
                    cfg[k][int(_k)] = cfg[k].pop(_k)
    return cfg

def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res

def online_agg_corr(agg_res, res, step):
    if agg_res is None:
        agg_res = {}
    for k, v in res.items():
        if k.startswith('l2_loss'):
            if k not in agg_res:
                agg_res[k] = []
            agg_res[k].append(np.mean(v))
        else:
            if k not in agg_res:
                agg_res[k] = v
            else:
                agg_res[k] = agg_res[k] + v

    return agg_res

def online_agg_append(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(v)
    return agg_res

def agg_func_pls(res):
    ret_dict = {}

    tmp_dict = {}
    for k, v in res.items():
        new_v = np.concatenate(v, axis=0)
        tmp_dict[k] = new_v

    rep_time = 5
    want_keys = ['V4_ave', 'IT_ave']

    for k, v in tmp_dict.items():
        if k in want_keys:
            continue
        for want_key in want_keys:
            for curr_rep in range(rep_time):
                from sklearn.cross_validation import KFold
                kf = KFold(tmp_dict['IT_ave'].shape[0], 4, shuffle = True, random_state = 0+curr_rep)

                neural_fea = tmp_dict[want_key]
                model_features_aftersub = v
                predict_fea = np.zeros(neural_fea.shape)
                for train, test in kf:
                    clf     = cross_decomposition.PLSRegression(n_components = 25, scale = False)

                    now_train_data      = model_features_aftersub[train, :]
                    now_train_label     = neural_fea[train, :]
                    now_test_data       = model_features_aftersub[test, :]
                    now_test_label      = neural_fea[test, :]

                    clf.fit(now_train_data, now_train_label)
                    new_test_label      = clf.predict(now_test_data)
                    predict_fea[test]   = new_test_label

                unit_score  = r2_score(neural_fea, predict_fea, multioutput='raw_values')
                print(np.sqrt(np.median(unit_score)), k, want_key)
                ret_dict['rep_%i/%s/%s' % (curr_rep, want_key, k)] = unit_score

                neuron_num = neural_fea.shape[1]
                corr_value = np.zeros(neuron_num)
                for which_neuron in range(neuron_num):
                    curr_corr = pearsonr(neural_fea[:, which_neuron], predict_fea[:, which_neuron])
                    corr_value[which_neuron] = curr_corr[0]
                print(np.median(corr_value), k, want_key)
                ret_dict['corr/rep_%i/%s/%s' % (curr_rep, want_key, k)] = corr_value

    return ret_dict

def agg_func(res):
    new_res = {}
    for k,v in res.items():
        if k.startswith('l2_loss'):
            new_res[k] = np.mean(v)
            _, curr_neuron, curr_node = k.split('/')

            n = res['corr/number']
            x_sum = res['corr/%s/sum' % curr_neuron]
            y_sum = res['corr/%s/%s/sum' % (curr_neuron, curr_node)]
            xy_sum = res['corr/%s/%s/mult_sum' % (curr_neuron, curr_node)]
            x2_sum = res['corr/%s/square_sum' % curr_neuron]
            y2_sum = res['corr/%s/%s/square_sum' % (curr_neuron, curr_node)]

            numerator = n*xy_sum - x_sum*y_sum
            denominator = np.sqrt((n*x2_sum - x_sum**2) * (n*y2_sum - y_sum**2)) + 1e-4

            new_res['corr/%s/%s' % (curr_neuron, curr_node)] = numerator/denominator

    return new_res

# 1 2 3 
# 4   5
# 6 7 8

def pos2lbl(pos):

    if pos == (0, 1):
        return 1
    elif pos == (1, 0):
        return 4
    elif pos == (1, 1):
        return 2
    elif pos == (0, -1):
        return 6
    elif pos == (-1, 0):
        return 3
    elif pos == (-1, -1):
        return 5
    elif pos == (-1, 1):
        return 0
    elif pos == (1, -1):
        return 7

def ab_to_Q(ab_images, path='pts_in_hull.npy', soft=True):
    pts_in_hull = tf.constant(np.load(path).T, dtype=tf.float32)
    diff  = tf.expand_dims(ab_images, axis=-1) - pts_in_hull
    distances = tf.reduce_sum(diff**2, axis=-2)
    if soft:
        T = 2.5e1
        return tf.nn.softmax(-distances/T)
    else:
        return tf.argmin(distances, axis=-1)

def Q_to_ab(Q, path='./pts_in_hull.npy', soft=True, is_logits=True):
    pts_in_hull = tf.constant(np.load(path), dtype=tf.float32)
    if soft:
        assert Q.shape.ndims == 4, Q.shape.ndims
        N, H, W, C = Q.get_shape().as_list()
        Q = tf.reshape(Q, [N * H * W, C])
        if is_logits:
            ab_images = tf.matmul(tf.nn.softmax(Q/0.38), pts_in_hull)
        else:
            ab_images = tf.matmul(Q, pts_in_hull)
        ab_images = tf.reshape(ab_images, [N, H, W] + ab_images.get_shape().as_list()[1:])
    else:
        if Q.shape.ndims == 4:
            Q = tf.argmax(Q, axis=-1)
        assert Q.shape.ndims == 3, Q.shape.ndims
        N, H, W = Q.get_shape().as_list()
        Q = tf.reshape(Q, [N*H*W])
        ab_images = tf.gather(pts_in_hull, Q)
        ab_images = tf.reshape(ab_images, [N, H, W] + ab_images.get_shape().as_list()[1:])
    return ab_images

def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        #srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))

def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        #lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))
