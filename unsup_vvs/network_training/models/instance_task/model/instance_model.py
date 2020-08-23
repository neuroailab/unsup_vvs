from __future__ import division, print_function, absolute_import
import os, sys
import json
import numpy as np
import tensorflow as tf
import copy
import pdb
from collections import OrderedDict

from . import resnet_model
from . import alexnet_model
from . import vggnet_model
from .memory_bank import MemoryBank
from .self_loss import get_selfloss, assert_shape, DATA_LEN_IMAGENET_FULL
from .resnet_th_preprocessing import ColorNormalize, ApplySobel, RGBtoGray
from .resnet_th_preprocessing import GrayNormalize, LNormalize, RGBtoLab
from .cluster_nn import NNClustering
from .cluster_lbl import LabelClustering


def flatten(layer_out):
    curr_shape = layer_out.get_shape().as_list()
    if len(curr_shape) > 2:
        layer_out = tf.reshape(layer_out, [curr_shape[0], -1])
    return layer_out


def get_alexnet_all_layers(all_layers, get_all_layers):
    if get_all_layers == 'default' or get_all_layers is None:
        keys = ['pool1', 'pool2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7']
    elif get_all_layers == 'conv_all':
        keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
    elif get_all_layers == 'conv5-avg':
        keys = ['conv5']
    else:
        keys = get_all_layers.split(',')

    output_dict = OrderedDict()
    for each_key in keys:
        for layer_name, layer_out in all_layers.items():
            if each_key in layer_name:
                if get_all_layers == 'conv5-avg':
                    layer_out = tf.nn.avg_pool(
                            layer_out,
                            ksize=[1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
                layer_out = flatten(layer_out)
                output_dict[each_key] = layer_out
                break
    return output_dict


def get_resnet_all_layers(ending_points, get_all_layers):
    ending_dict = OrderedDict()
    get_all_layers = get_all_layers.split(',')
    for idx, layer_out in enumerate(ending_points):
        if 'all_spatial' not in get_all_layers:
            if str(idx) in get_all_layers:
                layer_out = flatten(layer_out)
                ending_dict[str(idx)] = layer_out
        else:
            layer_out = tf.transpose(layer_out, [0,2,3,1])
            layer_name = 'encode_{layer_idx}'.format(layer_idx = idx+1)
            ending_dict[layer_name] = layer_out
    return ending_dict


def resnet_embedding(img_batch, dtype=tf.float32,
                     data_format=None, train=False,
                     resnet_size=18,
                     model_type='resnet',
                     resnet_version=resnet_model.DEFAULT_VERSION,
                     input_mode='rgb',
                     get_all_layers=None,
                     skip_final_dense=False
):
    normalize_func = {
        'rgb': ColorNormalize,
        'sobel': ColorNormalize,
        'gray': GrayNormalize,
        'L': LNormalize
    }

    if input_mode not in ['gray', 'rgb', 'L', 'sobel']:
        raise ValueError('Input mode must be one of [gray, rgb, L, sobel]')

    if input_mode == 'gray':
        img_batch = RGBtoGray(img_batch)

    image = tf.cast(img_batch, tf.float32)
    image = tf.div(image, tf.constant(255, dtype=tf.float32))

    if input_mode == 'L':
        image = RGBtoLab(image)
        image = tf.expand_dims(image[..., 0], -1)  # use the L channel only

    image = tf.map_fn(normalize_func[input_mode], image)

    if input_mode == 'sobel':
        image = ApplySobel(image)
        image = tf.squeeze(image)

    if model_type == 'resnet':
        model = resnet_model.ImagenetModel(
            resnet_size, data_format,
            resnet_version=resnet_version,
            dtype=dtype)

        if skip_final_dense and get_all_layers is None:
            return model(image, train, skip_final_dense=True)

        if skip_final_dense and get_all_layers.startswith('Mid-'):
            mid_layer_units = get_all_layers[4:]
            mid_layer_units = mid_layer_units.split(',')
            mid_layer_units = [int(each_unit) for each_unit in mid_layer_units]

            resnet_output = model(image, train, skip_final_dense=True)
            all_mid_layers = OrderedDict()
            with tf.variable_scope('instance', reuse=tf.AUTO_REUSE):
                init_builder = tf.contrib.layers.variance_scaling_initializer()
                for each_mid_unit in mid_layer_units:
                    now_name = 'mid{units}'.format(units=each_mid_unit)
                    mid_output = tf.layers.dense(
                            inputs=resnet_output, units=each_mid_unit,
                            kernel_initializer=init_builder,
                            trainable=True,
                            name=now_name)
                    all_mid_layers[now_name] = mid_output
            return all_mid_layers

        if get_all_layers:
            final_dense, ending_points = model(
                    image, train, get_all_layers=get_all_layers)
            all_layers = get_resnet_all_layers(ending_points, get_all_layers)
            all_layers['final_dense'] = final_dense
            return all_layers

        model_out = model(image, train, skip_final_dense=False)
    elif model_type == 'alexnet':
        model_out, _ = alexnet_model.alexnet_v2(
                image, is_training=train,
                num_classes=128)
    elif model_type == 'alexnet_bn':
        model_out, _ = alexnet_model.alexnet_v2_with_bn(
                image, is_training=train,
                num_classes=128)
    elif model_type == 'alexnet_bn_no_drop':
        model_out, all_layers = alexnet_model.alexnet_v2_with_bn_no_drop(
                image, is_training=train,
                num_classes=128)
        if get_all_layers or skip_final_dense:
            all_layers = get_alexnet_all_layers(all_layers, get_all_layers)
            if get_all_layers:
                return all_layers
            else:
                return all_layers['fc6']
    elif model_type == 'vggnet':
        model_out, _ = vggnet_model.vgg_16(
                image, is_training=train,
                num_classes=128, with_bn=True,
                dropout_keep_prob=0)
    elif model_type == 'convrnn':
        from . import convrnn_model
        model_out = convrnn_model.convrnn_model(image, train=train)
    else:
        raise ValueError('Model type not supported!')

    return tf.nn.l2_normalize(model_out, axis=1) # [bs, out_dim]


def repeat_1d_tensor(t, num_reps):
    ret = tf.tile(tf.expand_dims(t, axis=1), (1, num_reps))
    return ret


class InstanceModel(object):
    def __init__(self,
                 inputs, output,
                 memory_bank,
                 instance_k=4096,
                 instance_t=0.07,
                 instance_m=0.5,
                 multi_type='separate',
                 **kwargs):
        self.inputs = inputs
        self.embed_output = output
        self.batch_size, self.out_dim = self.embed_output.get_shape().as_list()
        self.memory_bank = memory_bank
        self.multi_type = multi_type

        self.instance_data_len = memory_bank.size
        self.instance_k = instance_k
        self.instance_t = instance_t
        self.instance_m = instance_m

    def _softmax(self, dot_prods):
        instance_Z = tf.constant(
            2876934.2 / 1281167 * self.instance_data_len,
            dtype=tf.float32)
        return tf.exp(dot_prods / self.instance_t) / instance_Z

    def compute_data_prob(self, selfloss):
        data_indx = self.inputs['index']
        logits = selfloss.get_closeness(data_indx, self.embed_output)
        return self._softmax(logits)

    def compute_noise_prob(self):
        noise_indx = tf.random_uniform(
            shape=(self.batch_size, self.instance_k),
            minval=0,
            maxval=self.instance_data_len,
            dtype=tf.int64)
        noise_probs = self._softmax(
            self.memory_bank.get_dot_products(self.embed_output, noise_indx))
        return noise_probs

    def updated_new_data_memory(self):
        data_indx = self.inputs['index'] # [bs]
        data_memory = self.memory_bank.at_idxs(data_indx)
        new_data_memory = (data_memory * self.instance_m
                           + (1 - self.instance_m) * self.embed_output)
        return tf.nn.l2_normalize(new_data_memory, axis=1)

    def __get_lbl_equal(self, each_k_idx, cluster_labels, top_idxs, k):
        batch_labels = tf.gather(
                cluster_labels[each_k_idx], 
                self.inputs['index'])
        if k > 0:
            top_cluster_labels = tf.gather(cluster_labels[each_k_idx], top_idxs)
            batch_labels = repeat_1d_tensor(batch_labels, k)
            curr_equal = tf.equal(batch_labels, top_cluster_labels)
        else:
            curr_equal = tf.equal(
                    tf.expand_dims(batch_labels, axis=1), 
                    tf.expand_dims(cluster_labels[each_k_idx], axis=0))
        return curr_equal

    def __get_prob_from_equal(self, curr_equal, exponents):
        probs = tf.reduce_sum(
            tf.where(
                curr_equal,
                x=exponents, y=tf.zeros_like(exponents),
            ), axis=1)
        probs /= tf.reduce_sum(exponents, axis=1)
        return probs

    def get_cluster_classification_loss(
            self, cluster_labels, 
            k=None, nns_domain=None):
        if not k:
            k = self.instance_k
        # ignore all but the top k nearest examples
        all_dps = self.memory_bank.get_all_dot_products(self.embed_output)
        top_dps, top_idxs = tf.nn.top_k(all_dps, k=k, sorted=False)
        if k > 0:
            exponents = self._softmax(top_dps)
        else:
            exponents = self._softmax(all_dps)

        no_kmeans = cluster_labels.get_shape().as_list()[0]
        if self.multi_type == 'separate':
            all_probs = []
            for each_k_idx in range(no_kmeans):
                curr_equal = self.__get_lbl_equal(
                        each_k_idx, cluster_labels, top_idxs, k)
                probs = self.__get_prob_from_equal(curr_equal, exponents)
                all_probs.append(probs)
            all_probs = sum(all_probs) / no_kmeans
        elif self.multi_type in ['or', 'and']:
            all_equal = None
            for each_k_idx in range(no_kmeans):
                curr_equal = self.__get_lbl_equal(
                        each_k_idx, cluster_labels, top_idxs, k)

                if all_equal is None:
                    all_equal = curr_equal
                else:
                    if self.multi_type == 'or':
                        all_equal = tf.logical_or(all_equal, curr_equal)
                    else:
                        all_equal = tf.logical_and(all_equal, curr_equal)
            probs = self.__get_prob_from_equal(all_equal, exponents)
        else:
            raise NotImplementedError("Unsupported multi_type")

        assert_shape(probs, [self.batch_size])
        loss = -tf.reduce_mean(tf.log(probs + 1e-7))
        new_nns = self.inputs['index']

        if nns_domain is not None:
            part_dps = tf.gather(all_dps, nns_domain, axis=1)
            _, new_nns = tf.nn.top_k(part_dps, k=1, sorted=False)
            new_nns = tf.cast(new_nns[:, 0], tf.int64)
        return (loss, loss, loss), new_nns

    def get_losses(self, data_prob, noise_prob):
        assert_shape(data_prob, [self.batch_size])
        assert_shape(noise_prob, [self.batch_size, self.instance_k])

        base_prob = 1.0 / self.instance_data_len
        eps = 1e-7
        ## Pmt
        data_div = data_prob + (self.instance_k*base_prob + eps)
        ln_data = tf.log(data_prob / data_div)
        ## Pon
        noise_div = noise_prob + (self.instance_k*base_prob + eps)
        ln_noise = tf.log((self.instance_k*base_prob) / noise_div)

        curr_loss = -(tf.reduce_sum(ln_data) \
                      + tf.reduce_sum(ln_noise)) / self.batch_size
        return curr_loss, \
            -tf.reduce_sum(ln_data)/self.batch_size, \
            -tf.reduce_sum(ln_noise)/self.batch_size


def build_output(
        inputs, train, 
        resnet_size=18,
        model_type='resnet',
        clstr_path=None,
        cluster_method='NN', kmeans_k=[10000],
        multi_type='separate',
        **kwargs):
    # This will be stored in the db
    logged_cfg = {'kwargs': kwargs}

    data_len = kwargs.get('instance_data_len', DATA_LEN_IMAGENET_FULL)
    with tf.variable_scope('instance', reuse=tf.AUTO_REUSE):
        all_labels = tf.get_variable(
            'all_labels',
            initializer=tf.zeros_initializer,
            shape=(data_len,),
            trainable=False,
            dtype=tf.int64,
        )
        # TODO: hard-coded output dimension 128
        memory_bank = MemoryBank(data_len, 128)

        nearest_neighbors = tf.get_variable(
            'nearest_neighbors',
            initializer=tf.range(data_len, dtype=tf.int64),
            trainable=False, dtype=tf.int64,
        )

        lbl_init_values = tf.range(data_len, dtype=tf.int64)
        no_kmeans_k = len(kmeans_k)
        lbl_init_values = tf.tile(
                tf.expand_dims(lbl_init_values, axis=0),
                [no_kmeans_k, 1])
        cluster_labels = tf.get_variable(
            'cluster_labels',
            initializer=lbl_init_values,
            trainable=False, dtype=tf.int64,
        )

    output = resnet_embedding(
            inputs['image'], train=train, 
            resnet_size=resnet_size,
            model_type=model_type)

    if not train:
        all_dist = memory_bank.get_all_dot_products(output)
        return [all_dist, all_labels], logged_cfg
    model_class = InstanceModel(
        inputs=inputs, output=output,
        memory_bank=memory_bank,
        multi_type=multi_type,
        **kwargs)

    nns_domain = None
    if cluster_method == 'NN':
        nn_clustering = NNClustering(10009, nearest_neighbors, cluster_labels)
    elif cluster_method == 'KM':
        from .cluster_km import Kmeans
        nn_clustering = Kmeans(kmeans_k, memory_bank, cluster_labels)
    elif cluster_method == 'SEMI':
        nn_clustering = LabelClustering(
                clstr_path, memory_bank, cluster_labels,
                nearest_neighbors)
        nns_domain = nn_clustering.label_index

    losses, new_nns = model_class.get_cluster_classification_loss(
            cluster_labels, nns_domain = nns_domain)

    new_data_memory = model_class.updated_new_data_memory()
    loss, loss_model, loss_noise = losses
    return {
        "losses": [loss, loss_model, loss_noise],
        "data_indx": inputs['index'],
        "memory_bank": memory_bank.as_tensor(),
        "new_data_memory": new_data_memory,
        "nns": nearest_neighbors,
        "new_nns": new_nns,
        "all_labels": all_labels,
    }, logged_cfg, nn_clustering


def build_transfer_output(
        inputs, train, 
        resnet_size=18,
        model_type='resnet',
        get_all_layers=None, 
        num_classes=1000,
        **kwargs):
    # This will be stored in the db
    logged_cfg = {'kwargs': kwargs}

    resnet_output = resnet_embedding(
        inputs['image'],
        train=False,
        resnet_size=resnet_size,
        model_type=model_type,
        skip_final_dense=True,
        get_all_layers=get_all_layers)

    with tf.variable_scope('instance', reuse=tf.AUTO_REUSE):
        init_builder = tf.contrib.layers.variance_scaling_initializer()
        if not get_all_layers:
            class_output = tf.layers.dense(
                inputs=resnet_output, units=num_classes,
                kernel_initializer=init_builder,
                trainable=True,
                name='transfer_dense')
        else:
            class_output = OrderedDict()
            for key, curr_out in resnet_output.items():
                class_output[key] = tf.layers.dense(
                    inputs=curr_out, units=num_classes,
                    kernel_initializer=init_builder,
                    trainable=True,
                    name='transfer_dense_{name}'.format(name=key))

    def __get_loss_accuracy(curr_output):
        _, pred = tf.nn.top_k(curr_output, k=1)
        pred = tf.cast(tf.squeeze(pred), tf.int64)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(pred, inputs['label']), tf.float32)
        )

        one_hot_labels = tf.one_hot(inputs['label'], num_classes)
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, curr_output)
        return loss, accuracy
    if not get_all_layers:
        loss, accuracy = __get_loss_accuracy(class_output)
    else:
        loss = []
        accuracy = OrderedDict()
        for key, curr_out in class_output.items():
            curr_loss, curr_acc = __get_loss_accuracy(curr_out)
            loss.append(curr_loss)
            accuracy[key] = curr_acc
        loss = tf.reduce_sum(loss)

    if not train:
        return accuracy, logged_cfg
    return [loss, accuracy], logged_cfg
