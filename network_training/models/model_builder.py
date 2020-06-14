import numpy as np
import tensorflow as tf
import sys
import copy
import pdb
from collections import OrderedDict
import tfutils
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
try:
    import pickle
except:
    import cPickle as pickle

import models.rp_col_utils as rc_utils
from models.rp_col_utils import rgb_to_lab
from models.mean_teacher_utils import ema_variable_scope, name_variable_scope
from models.model_blocks import NoramlNetfromConv
from models.config_parser import ConfigParser, get_network_cfg
from models.instance_header import InstanceHeader, InstClstrHeader, LAHeader
from models.instance_task.model.resnet_th_preprocessing import ApplySobel
import models.cpc_utils as cpc_utils


DATASET_PREFIX_LIST = [
        'scenenet', 'pbrnet', 'imagenet', 'imagenet_un', 
        'coco', 'place', 'kinetics', 'rp', 'colorization',
        'rp_imagenet', 'col_imagenet', 'imagenet_branch2',
        'saycam']
MEAN_RGB = [0.485, 0.456, 0.406] 
STD_RGB = [0.229, 0.224, 0.225]


def color_denormalize(image):
    transpose_flag = image.get_shape().as_list()[-1] != 3
    if transpose_flag:
        image = tf.transpose(image, [0, 2, 3, 1])

    imagenet_mean = np.array(MEAN_RGB, dtype=np.float32)
    imagenet_std = np.array(STD_RGB, dtype=np.float32)
    image = image * imagenet_std + imagenet_mean

    if transpose_flag:
        image = tf.transpose(image, [0, 3, 1, 2])
    return image


def color_normalize(image):
    transpose_flag = image.get_shape().as_list()[-1] != 3
    if transpose_flag:
        image = tf.transpose(image, [0, 2, 3, 1])

    imagenet_mean = np.array(MEAN_RGB, dtype=np.float32)
    imagenet_std = np.array(STD_RGB, dtype=np.float32)
    image = (image - imagenet_mean) / imagenet_std

    if transpose_flag:
        image = tf.transpose(image, [0, 3, 1, 2])
    return image


class ModelBuilder(object):
    """
    Build networks according to arguments and configs
    """
    def __init__(self, args, cfg_dataset):
        self.args = args
        self.cfg = ConfigParser(get_network_cfg(args))
        self.cfg_dataset = cfg_dataset
        self.any_la = False
        self.save_layer_middle_output = False

    def __need_dataset(self):
        return self.cfg_dataset.get(self.dataset_prefix, 0) == 1

    def __need_inst_clstr(self):
        str_inst_clstr = '{prefix}_instance_cluster'.format(
                prefix=self.dataset_prefix)
        return self.cfg_dataset.get(str_inst_clstr, 0) == 1

    def __need_LA(self):
        str_la = '{prefix}_LA'.format(prefix=self.dataset_prefix)
        return self.cfg_dataset.get(str_la, 0) == 1

    def __need_cpc(self):
        str_cpc = '{prefix}_cpc'.format(prefix=self.dataset_prefix)
        return self.cfg_dataset.get(str_cpc, 0) == 1

    def __get_LA_kmeans(self):
        str_LA_kmeans = '{prefix}_LA_kmeans'.format(prefix=self.dataset_prefix)
        LA_kmeans = self.cfg_dataset.get(str_LA_kmeans)
        LA_kmeans = LA_kmeans.split(',')
        LA_kmeans = [int(each_k) for each_k in LA_kmeans]
        return LA_kmeans

    def __need_instance(self):
        str_instance = '{prefix}_instance'.format(
                prefix=self.dataset_prefix)
        return self.cfg_dataset.get(str_instance, 0) == 1

    def __need_semi_clstr(self):
        str_semi_clstr = '{prefix}_semi_cluster'.format(
                prefix=self.dataset_prefix)
        return self.cfg_dataset.get(str_semi_clstr, 0) == 1

    def __need_mt_clstr(self, dataset_prefix=None):
        dataset_prefix = dataset_prefix or self.dataset_prefix
        str_mt_clstr = '{prefix}_mt_clstr'.format(
                prefix=dataset_prefix)
        return self.cfg_dataset.get(str_mt_clstr, 0) == 1

    def __need_any_mt(self):
        need_mt = False
        for dataset_prefix in DATASET_PREFIX_LIST:
            need_mt = need_mt or self.__need_mt_clstr(dataset_prefix)
        return need_mt

    def __need_image_dim_cmprss(self):
        return self.dataset_prefix.startswith('rp_')

    def __image_dim_cmprss(self, image_dataset):
        org_shape = image_dataset.get_shape().as_list()
        image_dataset = tf.reshape(image_dataset, [-1] + org_shape[-3:])
        return image_dataset

    def _get_and_color_norm_image(self, inputs, now_input_name):
        color_norm = self.args.color_norm
        tpu_task = self.args.tpu_task
        do_prep = not self.args.no_prep

        if isinstance(inputs, dict):
            image_dataset = tf.cast(inputs[now_input_name], tf.float32)
        else:
            image_dataset = tf.cast(inputs, tf.float32)

        if self.__need_image_dim_cmprss():
            image_dataset = self.__image_dim_cmprss(image_dataset)

        if tpu_task:
            if color_norm == 1:
                image_dataset = color_normalize(image_dataset)
        else:
            if do_prep:
                image_dataset = tf.div(
                        image_dataset, tf.constant(255, dtype=tf.float32))
                image_dataset = color_normalize(image_dataset)

        if self.__need_cpc():
            image_dataset = cpc_utils.image_preprocess(image_dataset)
        return image_dataset

    def __get_reuse_vars(self):
        self.var_name = self.cfg.get_var_name()
        self.var_offset = self.cfg.get_var_offset()
        self.reuse_name = '%s_reuse' % self.var_name
        self.reuse_flag = self.reuse_dict.get(self.reuse_name, None)

        curr_batch_name = '_%s' % self.dataset_prefix
        self.curr_batch_name = curr_batch_name + self.cfg.get_bn_var_name()
        self.curr_batch_name_indict \
                = '%s_bn_%s' % (self.var_name, self.curr_batch_name)
        self.reuse_batch = self.reuse_dict.get(self.curr_batch_name_indict, None)
        if self.dataset_prefix == 'nyuv2':
            self.curr_batch_name = '_%s' % 'pbrnet'
            reuse_batch = True
        if self.args.ignorebname_new:
            self.curr_batch_name = ''
            self.reuse_batch = self.reuse_flag

        if self.args.add_batchname is not None:
            self.curr_batch_name = self.args.add_batchname

        if self.args.combine_tpu_flag == 1 or self.args.tpu_task:
            self.reuse_flag = tf.AUTO_REUSE
            self.reuse_batch = tf.AUTO_REUSE

    def __set_reuse_vars(self):
        self.reuse_dict[self.reuse_name] = True
        self.reuse_dict[self.curr_batch_name_indict] = True

    def __get_mudule_inputs(self, image_dataset):
        input_now = image_dataset
        new_input_key = self.cfg.get_input_from_other_modules()

        self.valid_flag = True
        if new_input_key:
            assert new_input_key in self.all_out_dict, \
                    "Input nodes not built yet for module %s!" \
                    % self.module_name
            input_now = self.all_out_dict[new_input_key]
            self.valid_flag = False
        self.m.output = input_now

    def init_model_block_class(self):
        if self.args.seed == None:
            seed = 0
        else:
            seed = self.args.seed
        self.m = NoramlNetfromConv(
                seed=seed,
                global_weight_decay=self.args.global_weight_decay,
                enable_bn_weight_decay=self.args.enable_bn_weight_decay,
                )

    def __reset_model_block_class(self):
        self.m = None

    def __add_bypass_add(self, bypass_add):
        raise NotImplementedError

    def __add_bypass_light(self, add_bypass):
        for bypass_layer_name in add_bypass:
            all_out_dict = self.all_out_dict
            assert bypass_layer_name in all_out_dict, \
                    "Node %s not built yet for network %s!" \
                    % (bypass_layer_name, key_want)
            bypass_layer = all_out_dict[bypass_layer_name]
            self.m.add_bypass(bypass_layer)

    def __get_general_kwargs(self):
        general_kwargs = {
                'weight_decay': self.args.weight_decay, 
                'init': self.args.init_type,
                'stddev': self.args.init_stddev, 
                'train': self.train, 
                'reuse_flag': self.reuse_flag,
                'batch_name': self.curr_batch_name,
                'batch_reuse': self.reuse_batch,
                'sm_bn_trainable': self.args.sm_bn_trainable,
                }
        return general_kwargs

    def __get_resBlock_kwargs(self):
        kwargs = self.__get_general_kwargs()
        trainable_kwargs = self.cfg.get_resBlock_trainable_settings()
        kwargs.update(trainable_kwargs)
        return kwargs

    def __add_one_res_block(self):
        assert self.args.sm_resnetv2 == 0 and self.args.sm_resnetv2_1 == 0, \
                "Resnet V2 or V2_1 not supported yet"
        kwargs = self.__get_resBlock_kwargs()
        self.m.resblock(
                conv_settings=self.cfg.get_resBlock_conv_settings(),
                bias=0, 
                padding='SAME',
                **kwargs)

    def __get_conv_kwargs(self):
        conv_kwargs, conv_config = self.cfg.get_conv_kwargs()

        # Conv settings related to parameters not specified in the config
        trans_out_shape = None
        conv_upsample = conv_config.get('upsample', None)
        if conv_upsample:
            trans_out_shape = self.m.output.get_shape().as_list()
            trans_out_shape[1] = conv_upsample * trans_out_shape[1]
            trans_out_shape[2] = conv_upsample * trans_out_shape[2]
            trans_out_shape[3] = conv_config['num_filters']
        conv_kwargs['trans_out_shape'] = trans_out_shape

        padding = 'SAME'
        if self.valid_flag:
            padding = 'VALID'
            self.valid_flag = False
        padding = conv_config.get("conv_padding", None) or padding
        conv_kwargs['padding'] = padding

        general_kwargs = self.__get_general_kwargs()
        general_kwargs.update(conv_kwargs)
        return general_kwargs

    def __add_one_conv(self, curr_reuse_name):
        conv_kwargs = self.__get_conv_kwargs()
        self.m.conv(reuse_name=curr_reuse_name, **conv_kwargs)

    def __add_unpool(self):
        unpool_scale = self.cfg.get_unpool_scale()
        self.m.resize_images_scale(unpool_scale)

    def __get_fc_kwargs(self):
        fc_config = self.cfg.get_fc_config()
        fc_kwargs = {
                'init': self.args.init_type,
                'weight_decay': self.args.weight_decay,
                }
        fc_kwargs['out_shape'] = fc_config["num_features"]

        as_output = fc_config.get("output", None)
        if as_output:
            fc_kwargs['activation'] = None
            fc_kwargs['bias'] = 0
            fc_kwargs['dropout'] = None
        else:
            fc_kwargs['bias'] = .1
            fc_kwargs['dropout'] \
                    = None if not self.train else fc_config.get("dropout", 0.5)
        return fc_kwargs

    def __add_fc(self):
        fc_kwargs = self.__get_fc_kwargs()
        self.m.fc(**fc_kwargs)

    def __add_pool(self):
        pool_config = self.cfg.get_pool_config()
        pfs = pool_config['filter_size']
        ps = pool_config['stride']
        pool_type = pool_config.get('type', 'max')
        if pool_type == 'max':
            pfunc = 'maxpool'
        elif pool_type == 'avg':
            pfunc = 'avgpool'
        pool_padding = pool_config.get('padding', 'SAME')
        self.m.pool(pfs, ps, pfunc=pfunc, padding=pool_padding)

    def __add_upproj(self):
        self.m.upprojection(
                up_settings=self.cfg.get_upproj_settings(),
                weight_decay=self.args.weight_decay, 
                bias=0,
                init=self.args.init_type, stddev=self.args.init_stddev,
                train=self.train,
                reuse_flag=self.reuse_flag,
                batch_name=self.curr_batch_name,
                batch_reuse=self.reuse_batch,
                )

    def __add_ae_head(self):
        fc_kwargs = {
                'init': self.args.init_type,
                'weight_decay': self.args.weight_decay,
                }
        fc_kwargs['bias'] = .1
        fc_kwargs['dropout'] = None
        self.m.ae_head(dimension=self.cfg.get_ae_head_dim(), **fc_kwargs)

    def __build_one_layer(self, curr_layer):
        self.cfg.set_curr_layer(curr_layer)

        layer_name = "%s_%i" % (self.module_name, curr_layer)
        curr_reuse_name = '%s%i' % (self.var_name, curr_layer + self.var_offset)

        add_bypass_add = self.cfg.get_bypass_add()
        if add_bypass_add:
            self.__add_bypass_add(add_bypass_add)

        add_bypass_light = self.cfg.get_bypass_light()
        if add_bypass_light:
            self.__add_bypass_light(add_bypass_light)

        whether_res = self.cfg.get_whether_resBlock()
        if whether_res:
            with tf.variable_scope(curr_reuse_name):
                self.__add_one_res_block()

        whether_conv = self.cfg.whether_do_conv()
        if whether_conv:
            self.__add_one_conv(curr_reuse_name)
            if self.save_layer_middle_output:
                self.all_out_dict[layer_name + '.conv'] = self.m.output

        whether_unpool = self.cfg.get_whether_unpool()
        if whether_unpool:
            self.__add_unpool()

        whether_fc = self.cfg.get_whether_fc()
        if whether_fc:
            with tf.variable_scope(curr_reuse_name, reuse=self.reuse_flag):
                self.__add_fc()

        whether_pool = self.cfg.get_whether_pool()
        if whether_pool:
            self.__add_pool()

        whether_upproj = self.cfg.get_whether_upproj()
        if whether_upproj:
            with tf.variable_scope(curr_reuse_name):
                self.__add_upproj()

        whether_bn = self.cfg.get_whether_bn()
        if whether_bn:
            with tf.variable_scope(
                    '%s_bn%i%s' \
                            % (self.var_name, 
                               curr_layer + self.var_offset, 
                               self.curr_batch_name), 
                    reuse=self.reuse_batch):
                self.m.tpu_batchnorm(
                        self.train, 
                        sm_bn_trainable=self.args.sm_bn_trainable)

        whether_ae_head = self.cfg.get_whether_ae_head()
        if whether_ae_head:
            self.__add_ae_head()

        self.all_out_dict[layer_name] = self.m.output

    def __set_current_module(self, module_name):
        self.module_name = module_name
        self.cfg.set_current_module(module_name)

    def __build_convrnn_module(self, image_dataset):
        from convrnn_model import convrnn_model_func
        convrnn_outputs = convrnn_model_func(
                inputs={'images': image_dataset},
                **self.cfg.get_convrnn_params())
        self.all_out_dict.update(convrnn_outputs)

    def build_one_module(self, module_name, image_dataset):
        self.__set_current_module(module_name)

        if self.cfg.get_convrnn_params() is not None:
            self.__build_convrnn_module(image_dataset)
            return

        self.__get_mudule_inputs(image_dataset)
        self.__get_reuse_vars()

        with tf.contrib.framework.arg_scope(
                [self.m.conv], 
                init=self.args.init_type,
                stddev=self.args.init_stddev, 
                bias=0, activation='relu'):
            module_depth = self.cfg.get_module_depth()

            for curr_layer in range(1, module_depth + 1):
                self.__build_one_layer(curr_layer)

        as_output = self.cfg.get_as_output()
        if as_output:
            self.outputs_dataset[module_name] = self.m.output

        self.__set_reuse_vars()

    def __get_instance_embedding(self):
        last_output = list(self.outputs_dataset.values())[-1]
        embedding = tf.nn.l2_normalize(last_output, axis=1)
        return embedding

    def __build_instance_head(self, inputs):
        embedding = self.__get_instance_embedding()

        var_scope_memory = self.dataset_prefix
        if self.dataset_prefix == 'imagenet_un':
            var_scope_memory = 'imagenet'

        instance_builder = InstanceHeader(
                inputs, embedding, 
                var_scope_memory, self.dataset_prefix,
                tpu_task=self.args.tpu_task,
                instance_k=self.args.instance_k,
                instance_t=self.args.instance_t,
                instance_m=self.args.instance_m,
                instance_data_len=self.args.instance_data_len,
                instance_lbl_pkl=self.args.inst_lbl_pkl,
                )
        self.outputs_dataset['instance'] \
                = instance_builder.build_header(self.train)

    def __build_LA(self, inputs):
        self.any_la = True
        self.last_clustering_step = None
        embedding = self.__get_instance_embedding()
        var_scope_memory = '{name}_LA'.format(name=self.dataset_prefix)
        LA_kmeans = self.__get_LA_kmeans()
        LA_builder = LAHeader(
                inputs, embedding, 
                LA_kmeans, var_scope_memory, self.dataset_prefix,
                tpu_task=self.args.tpu_task,
                instance_k=self.args.instance_k,
                instance_t=self.args.instance_t,
                instance_m=self.args.instance_m,
                instance_data_len=self.args.instance_data_len,
                )
        if self.train:
            self.outputs_dataset['LA'], clustering \
                    = LA_builder.build_header(True)
            if getattr(self, 'LA_clusterings', None) is None:
                self.LA_clusterings = []
            self.LA_clusterings.append(clustering)
        else:
            self.outputs_dataset['LA'] = LA_builder.build_header(False)

    def __build_inst_clstr(self, inputs):
        embedding = self.__get_instance_embedding()

        var_scope_memory = '{name}_clstr'.format(name=self.dataset_prefix)
        inst_clstr_builder = InstClstrHeader(
                inputs, embedding, 
                self.args.inst_clstr_path,
                var_scope_memory, self.dataset_prefix,
                instance_k=self.args.instance_k,
                instance_t=self.args.instance_t,
                instance_m=self.args.instance_m)
        self.outputs_dataset['inst_clstr'] \
                = inst_clstr_builder.build_header(self.train)

    def __build_semi_clstr(self, inputs):
        embedding = self.__get_instance_embedding()

        if not self.args.semi_name_scope:
            var_scope_memory = '{name}_semi'.format(name=self.dataset_prefix)
        else:
            var_scope_memory = self.args.semi_name_scope
        inst_clstr_builder = InstClstrHeader(
                inputs, embedding, 
                self.args.semi_clstr_path,
                var_scope_memory, self.dataset_prefix,
                instance_k=self.args.instance_k,
                instance_t=self.args.instance_t,
                instance_m=self.args.instance_m,
                use_semi=True)
        self.outputs_dataset['semi_clstr'] \
                = inst_clstr_builder.build_header(self.train)

    def __need_rp(self):
        return self.dataset_prefix.startswith('rp_')

    def __build_rp(self):
        curr_output = self.m.output
        rp_input = rc_utils.build_rp_input(curr_output)
        self.build_one_module('rp_category', rp_input)

    def __build_cpc(self):
        features = self.outputs_dataset['encode']
        cpc_loss = cpc_utils.build_cpc_loss(features)
        self.outputs_dataset['cpc_loss'] = cpc_loss

    def _build_datasetnet(self, inputs, dataset_prefix):
        args = self.args
        now_input_name = 'image_%s' % dataset_prefix
        self.all_out_dict = {}
        self.dict_cache_filter = {}
        self.dataset_prefix = dataset_prefix
        self.outputs_dataset = OrderedDict()

        instance_task = self.args.instance_task or self.__need_instance()
        if (dataset_prefix == 'imagenet' and self.args.inst_cate_sep) \
                or (dataset_prefix == 'rp_imagenet'):
            instance_task = False
        if dataset_prefix != 'imagenet':
            instance_task = self.__need_instance()

        if self.__need_dataset() \
                and (args.tpu_task or now_input_name in inputs):
            image_dataset = self._get_and_color_norm_image(
                    inputs, now_input_name)
            module_order = self.cfg.get_dataset_order(dataset_prefix)

            for module_name in module_order:
                self.build_one_module(module_name, image_dataset)

            if instance_task or self.args.tpu_task=='instance_task':
                self.__build_instance_head(inputs)

            if self.__need_LA():
                self.__build_LA(inputs)

            if self.__need_inst_clstr():
                self.__build_inst_clstr(inputs)

            if self.__need_semi_clstr():
                self.__build_semi_clstr(inputs)

            if self.__need_rp():
                self.__build_rp()

            if self.__need_cpc():
                self.__build_cpc()

            if self.args.tpu_task and dataset_prefix == 'rp':
                raise NotImplementedError("RP on tpu not supported yet")
        return self.outputs_dataset

    def train_loop(self, sess, train_targets, **params):
        if self.any_la:
            global_step_vars = [v for v in tf.global_variables() \
                                if 'global_step' in v.name]
            assert len(global_step_vars) == 1
            global_step = sess.run(global_step_vars[0])

            # TODO: consider making this reclustering frequency a flag
            str_freq = '{prefix}_LA_freq'.format(prefix=self.dataset_prefix)
            freq = self.cfg_dataset.get(str_freq, 10010)
            if (self.last_clustering_step is None) \
                    or ((global_step - self.last_clustering_step) % freq == 0):
                print("Recomputing clusters...")
                new_clust_labels = self.LA_clusterings[0].recompute_clusters(
                        sess)
                for clustering in self.LA_clusterings:
                    clustering.apply_clusters(sess, new_clust_labels)
                self.last_clustering_step = global_step
        return tfutils.defaults.train_loop(sess, train_targets, **params)

    def build(self, inputs, train=True, **kwargs):
        args = self.args

        self.all_outputs = OrderedDict()
        self.all_datasets_out_dict = {}
        self.reuse_dict = {}
        self.train = train
        self.init_model_block_class()

        mean_teacher = args.mean_teacher or self.__need_any_mt()

        if mean_teacher:
            self.all_outputs['primary'] = OrderedDict()
            self.all_outputs['ema'] = OrderedDict()

        for dataset_prefix in DATASET_PREFIX_LIST:
            if not mean_teacher:
                self.all_outputs[dataset_prefix] = self._build_datasetnet(
                        inputs, dataset_prefix=dataset_prefix)
                self.all_datasets_out_dict[dataset_prefix] = self.all_out_dict
            else: 
                with name_variable_scope("primary", 
                                         "primary", 
                                         reuse=tf.AUTO_REUSE) \
                        as (name_scope, var_scope):
                    self.all_outputs['primary'][dataset_prefix] \
                            = self._build_datasetnet(
                                    inputs, dataset_prefix=dataset_prefix)
                if dataset_prefix in ['imagenet', 'imagenet_un']:
                    with ema_variable_scope(
                            "ema", var_scope, 
                            decay=self.args.ema_decay, 
                            zero_debias=self.args.ema_zerodb, 
                            reuse=tf.AUTO_REUSE):
                        self.all_outputs['ema'][dataset_prefix] \
                                = self._build_datasetnet(
                                        inputs, dataset_prefix=dataset_prefix)

        #self._build_datasetnet(
        #        inputs, dataset_prefix='nyuv2')
        self.__reset_model_block_class()
        all_outputs = self.all_outputs
        if args.do_pca:
            all_outputs = self.all_datasets_out_dict
        self.all_outputs = None
        self.outputs_dataset = None
        self.all_out_dict = None
        if args.tpu_task=='cpc':
            cpc_loss = all_outputs['imagenet']['cpc_loss']
            cpc_loss = tf.expand_dims(cpc_loss, axis=0)
            batch_size = args.batchsize // 8
            cpc_loss = tf.tile(cpc_loss, [batch_size])
            return cpc_loss
        else:
            return all_outputs, vars(args)

    def compute_pca(self, res):
        args = self.args
        layer_names = filter(
                lambda key: key.startswith('encode'),
                res[0]['imagenet'].keys(), 
                )
        all_pcas = {}
        for each_layer in tqdm(layer_names):
            _output = np.concatenate(
                    [_each_res['imagenet'][each_layer] for _each_res in res],
                    axis=0)
            pca = PCA(n_components=args.pca_n_components, random_state=0)
            pca.fit(_output.reshape((_output.shape[0], -1)))
            all_pcas[each_layer] = pca
        pca_save_path = os.path.join(
                args.pca_save_dir, 
                args.expId + '.pkl')
        pickle.dump(all_pcas, open(pca_save_path, 'wb'))
        return {'path': pca_save_path}
