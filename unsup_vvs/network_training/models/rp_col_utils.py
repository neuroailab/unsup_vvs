from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np
import tensorflow as tf
import copy
import pdb
import json

RP_POS_PAIRs = [
        (0,1),(0,-1),(1,0),(-1,0),
        (1,1),(-1,-1),(1,-1),(-1,1),
        (1,0),(-1,0),(0,1),(0,-1)]

RP_POS_HEADs = [
        (0,1), (1,0), (0,2), (2,0),
        (0,3), (3,0), (1,2), (2,1),
        (1,3), (3,1), (2,3), (3,2)]


def build_rp_input(curr_output):
    curr_shape = curr_output.get_shape().as_list()
    new_shape = [int(curr_shape[0] / 4), 4] + curr_shape[1:]
    curr_output = tf.reshape(curr_output, new_shape)
    all_heads = tf.unstack(curr_output, axis=1)
    for which_head in range(4):
        all_heads[which_head] = tf.reshape(
                all_heads[which_head], 
                [-1, 1, 1, 1, np.prod(new_shape[2:])])
    all_pairs = []
    for each_pos_head in RP_POS_HEADs:
        all_pairs.append(
                tf.concat(
                    [all_heads[each_pos_head[0]], all_heads[each_pos_head[1]]],
                    axis=-1))
    rp_input = tf.concat(all_pairs, axis=1)
    rp_input = tf.reshape(rp_input, [-1] + rp_input.get_shape().as_list()[2:])
    return rp_input


def get_rp_loss(cmprssed_output):
    output = tf.reshape(cmprssed_output, [-1, 12, 8])
    batch_size = output.get_shape().as_list()[0]
    all_labels = []
    for each_pair in RP_POS_PAIRs:
        one_hot_labels = tf.one_hot(pos2lbl(each_pair), 8)
        one_hot_labels = tf.reshape(one_hot_labels, [1, 1, 8])
        one_hot_labels = tf.tile(one_hot_labels, [batch_size, 1, 1])
        all_labels.append(one_hot_labels)
    all_labels = tf.concat(all_labels, axis=1)
    rp_loss = tf.losses.softmax_cross_entropy(
            logits=output, onehot_labels=all_labels)
    return rp_loss


def get_rp_top1(cmprssed_output):
    all_labels_ = []
    batch_size = int(cmprssed_output.get_shape().as_list()[0] / 12)
    for each_pair in RP_POS_PAIRs:
        labels_ = tf.constant(pos2lbl(each_pair), dtype=tf.int64)
        labels_ = tf.reshape(labels_, [1, 1])
        labels_ = tf.tile(labels_, [batch_size, 1])
        all_labels_.append(labels_)
    curr_label = tf.concat(all_labels_, axis=1)
    curr_label = tf.reshape(curr_label, [-1])
    curr_top1 = tf.nn.in_top_k(cmprssed_output, curr_label, 1)
    return curr_top1


def get_col_loss(pred_col, gt_col):
    num_col_cat = pred_col.get_shape().as_list()[-1]
    pred_col = tf.reshape(pred_col, [-1, num_col_cat])
    gt_col = tf.reshape(gt_col, [-1, num_col_cat])
    loss = tf.losses.softmax_cross_entropy(
            logits=pred_col, onehot_labels=gt_col)
    return loss


def get_col_top1(pred_col, gt_col):
    num_col_cat = pred_col.get_shape().as_list()[-1]
    pred_col = tf.reshape(pred_col, [-1, num_col_cat])
    gt_col = tf.argmax(gt_col, -1)
    gt_col = tf.reshape(gt_col, [-1])
    top_1 = tf.cast(tf.nn.in_top_k(pred_col, gt_col, 1), tf.float32)
    return top_1


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


def ab_to_Q(ab_images, path='./pts_in_hull.npy', soft=True, col_knn=False):
    print("ab_to_Q function:")
    pts_in_hull = tf.constant(np.load(path).T, dtype=tf.float32)
    #print("pts_in_hull:", pts_in_hull)
    diff = tf.expand_dims(ab_images, axis=-1) - pts_in_hull
    print("diff:", diff)
    distances = tf.reduce_sum(diff**2, axis=-2)
    print("distances:", distances)
    tpu_flag = 0 # tpus process images one by one
    if distances.shape.ndims == 3:
        distances = tf.expand_dims(distances, axis=0)
        tpu_flag = 1
    if col_knn:
        distances = tf.sqrt(distances)
        distances = -distances  # The lower of the distance, the better
        values, indices = tf.nn.top_k(distances, k=5, sorted=False) # 5 nearest neighbor
        print("values:", values)
        print("indices:", indices)
        print("distances2:", distances.get_shape().as_list())
        N, H, W, Q = distances.get_shape().as_list()

        weights = tf.exp(-(values**2)/(2*5.0**2)) # gaussian kernel
        print("weights:", weights)
        weights = weights / tf.expand_dims(tf.reduce_sum(weights, axis=-1), axis=-1) # scale, sum to 1

        Q_label = tf.constant([N * H * W, Q])
        weights = tf.reshape(weights, [-1])
        print("weights2:", weights)
        #indices = tf.expand_dims(indices)
        #indices = tf.cast(indices, tf.int32)
        indice1 = tf.range(N*H*W)
        indice1 = tf.reshape(indice1, [N*H*W, 1])
        indice1 = tf.tile(indice1, [1, 5])
        indice1 = tf.reshape(indice1, [1, -1])
        indice2 = tf.reshape(indices, [1, -1])
        indice = tf.concat([indice1, indice2], axis=0)
        indice = tf.transpose(indice, [1, 0])
        print("indice:", indice)
        Q_label = tf.scatter_nd(indice, weights, Q_label)
        Q_label = tf.reshape(Q_label, [N, H, W, Q])
    else:
        if soft:
            T = 2.5e1
            Q_label =  tf.nn.softmax(-distances/T)
        else:
            Q_label = tf.argmin(distances, axis=-1)
    
    print("Q_label:", Q_label)

    if tpu_flag:
        Q_label = tf.squeeze(Q_label, axis=0)

    return Q_label

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


def Q_to_ab(Q, path='./pts_in_hull.npy', soft=True, col_knn=False, is_logits=True):
    pts_in_hull = tf.constant(np.load(path), dtype=tf.float32)
    if soft or col_knn:
        assert Q.shape.ndims == 4, Q.shape.ndims
        N, H, W, C = Q.get_shape().as_list()
        Q = tf.reshape(Q, [N * H * W, C])
        if is_logits:
            ab_images = tf.matmul(tf.nn.softmax(Q), pts_in_hull)
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
