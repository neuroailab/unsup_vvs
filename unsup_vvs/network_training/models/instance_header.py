import os, sys
import json
import numpy as np
import tensorflow as tf
import copy
import pdb
from unsup_vvs.network_training.models.instance_task.model.memory_bank import MemoryBank
from unsup_vvs.network_training.models.instance_task.model.instance_model import InstanceModel
from unsup_vvs.network_training.models.instance_task.model.cluster_lbl import get_clstr_labels_and_index
from unsup_vvs.network_training.models.instance_task.model.self_loss import DATA_LEN_IMAGENET_FULL

INSTANCE_Z_DEFAULT = 2876934.2


def get_instance_softmax(dist, instance_t, instance_data_len):
    prob = tf.exp(dist / instance_t)
    instance_Z = tf.constant(
            INSTANCE_Z_DEFAULT / 1281167 * instance_data_len, 
            dtype=tf.float32)
    prob /= instance_Z
    return prob


class InstanceHeader(object):
    """
    Build the instance header
    """
    def __init__(
            self,
            inputs,
            embedding,
            var_scope_memory,
            dataset_prefix,
            tpu_task=None,
            instance_k=4096,
            instance_t=0.07,
            instance_m=0.5,
            instance_data_len=1281167,
            instance_lbl_pkl=None,
            ):
        self.instance_k = instance_k
        self.instance_t = instance_t
        self.instance_m = instance_m
        self.instance_data_len = instance_data_len
        self.instance_lbl_pkl = instance_lbl_pkl
        self.tpu_task = tpu_task

        self.embedding = embedding
        self.inputs = inputs
        self.var_scope_memory = var_scope_memory
        self.dataset_prefix = dataset_prefix

    def __define_memory_bank_all_labels(self):
        with tf.variable_scope(self.var_scope_memory, reuse=tf.AUTO_REUSE):
            ### Set the variable and initial values
            self.batch_size, self.out_dim = self.embedding.get_shape().as_list()
            model_seed = 0

            #### Get initial memory bank
            if self.tpu_task is None:
                mb_init = tf.random_uniform(
                        shape=(self.instance_data_len, self.out_dim),
                        seed=model_seed,
                        )
            else:
                mb_init = np.random.uniform(
                        size=(self.instance_data_len, self.out_dim))
                mb_init = mb_init.astype(np.float32)

            std_dev = 1. / np.sqrt(self.out_dim/3)
            mb_init = mb_init * (2*std_dev) - std_dev
            self.memory_bank = tf.get_variable(
                    'memory_bank', 
                    initializer=mb_init,
                    dtype=tf.float32,
                    trainable=False,
                    )

            #### Get initial all labels
            if self.instance_lbl_pkl is None:
                label_init = tf.zeros_initializer
                all_label_kwarg = {
                        'shape':(self.instance_data_len),
                        }
            else:
                label_init = cPickle.load(open(self.instance_lbl_pkl, 'r'))
                label_init = label_init.astype(np.int64)
                all_label_kwarg = {}
            self.all_labels = tf.get_variable(
                    'all_labels',
                    initializer=label_init,
                    trainable=False,
                    dtype=tf.int64,
                    **all_label_kwarg
                    )

    def __get_data_indx(self):
        index_name = 'index_%s' % self.dataset_prefix
        assert index_name in self.inputs, "Input should include index!"
        data_indx = self.inputs[index_name]
        return data_indx

    def __get_data_noise_prob(self):
        data_indx = self.__get_data_indx()
        self.data_indx = data_indx
        noise_indx = tf.random_uniform(
                shape=(self.batch_size, self.instance_k),
                minval=0,
                maxval=self.instance_data_len,
                dtype=tf.int64)
        # data_memory: [bs, out_dim]
        data_memory = tf.gather(self.memory_bank, data_indx, axis=0) 
        self.data_memory = data_memory
        # noise_memory [bs, k, out_dim]
        noise_memory = tf.reshape(
                tf.gather(self.memory_bank, noise_indx, axis=0),
                [self.batch_size, self.instance_k, self.out_dim]
                ) 
        ### Compute the data distance and noise distance
        curr_out_ext = tf.expand_dims(self.embedding, axis=1)
        data_dist = tf.reshape(
                tf.matmul(
                    curr_out_ext, 
                    tf.expand_dims(data_memory, axis=2)), 
                [self.batch_size]) # [bs]
        noise_dist = tf.squeeze(
                tf.matmul(
                    curr_out_ext, 
                    tf.transpose(noise_memory, [0,2,1])),
                axis=1) # [bs, k]

        data_prob = get_instance_softmax(
                data_dist, self.instance_t,
                self.instance_data_len)
        noise_prob = get_instance_softmax(
                noise_dist, self.instance_t,
                self.instance_data_len)
        return data_prob, noise_prob

    def __get_new_data_memory(self):
        new_data_memory \
                = self.data_memory * self.instance_m \
                + (1 - self.instance_m) * self.embedding
        new_data_memory = tf.nn.l2_normalize(
                new_data_memory, 
                axis=1)
        return new_data_memory

    def __memory_update_on_tpu(self):
        update_data_memory = self.new_data_memory - self.data_memory
        scatter_memory = tf.scatter_nd(
                tf.expand_dims(self.data_indx, axis=1),
                update_data_memory,
                shape=self.memory_bank.shape)
        # On tpu, collecting all updates on each tpu core
        scatter_memory = tf.contrib.tpu.cross_replica_sum(scatter_memory)
        mb_update_op = tf.assign_add(
                self.memory_bank, 
                scatter_memory,
                use_locking=False)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mb_update_op)

    def build_header(self, train):
        self.__define_memory_bank_all_labels()

        ret_outputs = {}
        if train:
            data_prob, noise_prob = self.__get_data_noise_prob()
            ret_outputs['data_prob'] = data_prob
            ret_outputs['noise_prob'] = noise_prob
            self.new_data_memory = self.__get_new_data_memory()

            if not self.tpu_task:
                ret_outputs['memory_bank'] = self.memory_bank
                ret_outputs['data_indx'] = self.data_indx
                ret_outputs['new_data_memory'] = self.new_data_memory
            else:
                self.__memory_update_on_tpu()

            ### Update the labels
            if self.instance_lbl_pkl is None:
                ret_outputs['all_labels'] = self.all_labels
        else:
            all_dist = tf.matmul(
                    self.embedding, 
                    tf.transpose(self.memory_bank, [1, 0])) # [bs, data_len]
            ret_outputs = [all_dist, self.all_labels]
        return ret_outputs


class InstClstrHeader(object):
    """
    Build header for instance cluster task
    """
    def __init__(
            self, inputs, embedding, 
            inst_clstr_path, var_scope_memory, dataset_prefix,
            use_semi=False, indx_for_clstr=None,
            **kwargs):
        self.embedding = embedding
        self.inst_clstr_path = inst_clstr_path
        self.var_scope_memory = var_scope_memory
        self.other_kwargs = kwargs
        self.dataset_prefix = dataset_prefix
        self.use_semi = use_semi

        if not use_semi:
            assert os.path.exists(inst_clstr_path), \
                    "Cluster label file doesn't exist!"
            self.clstr_labels = np.load(inst_clstr_path)
            self.data_len = len(self.clstr_labels)
        else:
            self.semi_clstr_labels, self.label_index \
                    = get_clstr_labels_and_index(
                            inst_clstr_path, indx_for_clstr)
            self.data_len = DATA_LEN_IMAGENET_FULL
            self.semi_clstr_labels += DATA_LEN_IMAGENET_FULL

        self.inputs = self.__get_new_inputs(inputs)

    def __get_new_inputs(self, inputs):
        index_name = 'index_%s' % self.dataset_prefix
        assert index_name in inputs, "Input should include index!"
        new_inputs = {'index': inputs[index_name]}
        return new_inputs

    def __define_memory_bank_all_labels(self):
        with tf.variable_scope(self.var_scope_memory, reuse=tf.AUTO_REUSE):
            ### Set the variable and initial values
            self.batch_size, self.out_dim = self.embedding.get_shape().as_list()
            model_seed = 0

            self.memory_bank = MemoryBank(self.data_len, self.out_dim)
            self.all_labels = tf.get_variable(
                    'all_labels',
                    initializer=tf.zeros_initializer,
                    trainable=False,
                    dtype=tf.int64,
                    shape=(self.data_len))

            if self.use_semi:
                self.clstr_labels = tf.get_variable(
                        'cluster_labels',
                        initializer=tf.range(self.data_len, dtype=tf.int64),
                        trainable=False, dtype=tf.int64,
                        )

    def build_header(self, train):
        self.__define_memory_bank_all_labels()

        if not train:
            all_dist = self.memory_bank.get_all_dot_products(self.embedding)
            return [all_dist, self.all_labels]

        instance_model = InstanceModel(
                inputs=self.inputs, output=self.embedding,
                memory_bank=self.memory_bank,
                **self.other_kwargs)

        nns_domain = None
        if self.use_semi:
            nns_domain = self.label_index
        losses, new_nns = instance_model.get_cluster_classification_loss(
                self.clstr_labels, nns_domain=nns_domain)
        new_data_memory = instance_model.updated_new_data_memory()
        ret_dict = {
                'loss': losses[0],
                'memory_bank': self.memory_bank.as_tensor(),
                "new_data_memory": new_data_memory,
                "all_labels": self.all_labels,
                }
        if self.use_semi:
            new_clstr_labels = tf.gather(
                    self.semi_clstr_labels, 
                    new_nns)
            ret_dict.update({
                "new_clstr_labels": new_clstr_labels,
                "clstr_labels": self.clstr_labels,
                })
        return ret_dict


class LAHeader(object):
    """
    Build header for local aggregation task
    """
    def __init__(
            self, inputs, embedding, 
            LA_kmeans, var_scope_memory, dataset_prefix,
            **kwargs):
        self.embedding = embedding
        self.LA_kmeans = LA_kmeans
        self.var_scope_memory = var_scope_memory
        self.dataset_prefix = dataset_prefix
        self.data_len = kwargs.get('instance_data_len', DATA_LEN_IMAGENET_FULL)
        self.other_kwargs = kwargs
        self.inputs = self.__get_new_inputs(inputs)

    def __get_new_inputs(self, inputs):
        index_name = 'index_%s' % self.dataset_prefix
        assert index_name in inputs, "Input should include index!"
        new_inputs = {'index': inputs[index_name]}
        return new_inputs

    def __define_memory_bank_all_labels(self):
        with tf.variable_scope(self.var_scope_memory, reuse=tf.AUTO_REUSE):
            ### Set the variable and initial values
            self.batch_size, self.out_dim = self.embedding.get_shape().as_list()
            model_seed = 0

            self.memory_bank = MemoryBank(self.data_len, self.out_dim)
            self.all_labels = tf.get_variable(
                    'all_labels',
                    initializer=tf.zeros_initializer,
                    trainable=False,
                    dtype=tf.int64,
                    shape=(self.data_len))

            lbl_init_values = tf.range(self.data_len, dtype=tf.int64)
            no_kmeans_k = len(self.LA_kmeans)
            lbl_init_values = tf.tile(
                    tf.expand_dims(lbl_init_values, axis=0),
                    [no_kmeans_k, 1])
            self.cluster_labels = tf.get_variable(
                'cluster_labels',
                initializer=lbl_init_values,
                trainable=False, dtype=tf.int64,
            )

    def build_header(self, train):
        self.__define_memory_bank_all_labels()

        if not train:
            all_dist = self.memory_bank.get_all_dot_products(self.embedding)
            return [all_dist, self.all_labels]

        instance_model = InstanceModel(
                inputs=self.inputs, output=self.embedding,
                memory_bank=self.memory_bank,
                multi_type='or',
                **self.other_kwargs)

        nns_domain = None
        losses, _ = instance_model.get_cluster_classification_loss(
                self.cluster_labels, nns_domain=nns_domain)
        new_data_memory = instance_model.updated_new_data_memory()
        ret_dict = {
                'loss': losses[0],
                'memory_bank': self.memory_bank.as_tensor(),
                "new_data_memory": new_data_memory,
                "all_labels": self.all_labels,
                'data_indx': self.inputs['index'],
                }
        from models.instance_task.model.cluster_km import Kmeans
        clustering = Kmeans(
                self.LA_kmeans, self.memory_bank, self.cluster_labels)
        return ret_dict, clustering
