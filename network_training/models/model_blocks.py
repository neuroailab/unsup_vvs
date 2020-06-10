from collections import OrderedDict

import numpy as np
import tensorflow as tf
import pdb
import os, sys


class ConvNet(object):
    """Basic implementation of ConvNet class compatible with tfutils.
    """

    def __init__(
            self, seed=None, 
            fixweights=False, global_weight_decay=None, 
            conv2d_data_format="NHWC",
            conv2d_with_bias=True,
            enable_bn_weight_decay=False,
            **kwargs):
        self.seed = seed
        self.output = None
        self._params = OrderedDict()
        self.default_trainable = True
        if fixweights:
            print('Will use random weights!')
            self.default_trainable = False
        self.global_weight_decay = global_weight_decay
        self.conv2d_data_format = conv2d_data_format
        self.conv2d_with_bias = conv2d_with_bias
        self.enable_bn_weight_decay = enable_bn_weight_decay

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        name = tf.get_variable_scope().name
        if name not in self._params:
            self._params[name] = OrderedDict()
        self._params[name][value['type']] = value

    @property
    def graph(self):
        return tf.get_default_graph().as_graph_def()

    def initializer(self, kind='xavier', stddev=0.01, init_file=None, init_keys=None):
        #print(kind)
        if kind == 'xavier':
            init = tf.contrib.layers.xavier_initializer(seed=self.seed)
        elif kind == 'trunc_norm':
            init = tf.truncated_normal_initializer(mean=0, stddev=stddev, seed=self.seed)
        elif kind == 'variance_scaling_initializer':
            #print('Using %s' % kind)
            init = tf.contrib.layers.variance_scaling_initializer(seed=self.seed)
        elif kind == 'from_file':
            # If we are initializing a pretrained model from a file, load the key from this file
            # Assumes a numpy .npz object
            # init_keys is going to be a dictionary mapping {'weight': weight_key,'bias':bias_key}
            params = np.load(init_file)
            init = {}
            init['weight'] = params[init_keys['weight']]
            init['bias'] = params[init_keys['bias']]
        elif kind == 'from_cached':
            # If we are initializing a pretrained model from a cached dict, load the key from this dict
            # init_keys is going to be a dictionary mapping {'weight': weight_tensor,'bias':bias_tensor}
            init = init_keys
        elif kind=='norm':
            init = tf.random_normal_initializer(mean=0, stddev=stddev, seed=self.seed)
        elif kind=='uniform':
            init = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, seed=self.seed)
        else:
            raise ValueError('Please provide an appropriate initialization '
                             'method: xavier or trunc_norm')
        return init

    @tf.contrib.framework.add_arg_scope
    def tpu_batchnorm(
            self, is_training, inputs = None, 
            decay = 0.997, epsilon = 1e-5, sm_bn_trainable = True):
        if inputs==None:
            inputs = self.output
        axis=-1
        if self.conv2d_data_format=='NCHW':
            axis=1
        beta_regularizer = None
        gamma_regularizer = None
        if self.enable_bn_weight_decay:
            weight_decay = self.global_weight_decay
            beta_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
            gamma_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

        self.output = tf.layers.batch_normalization(
                inputs=inputs, axis=axis,
                momentum=decay, epsilon=epsilon, 
                center=True, scale=True, 
                training=is_training, fused=True, 
                trainable=sm_bn_trainable,
                beta_regularizer=beta_regularizer,
                gamma_regularizer=gamma_regularizer,
                )
        return self.output

    @tf.contrib.framework.add_arg_scope
    def conv(self,
             out_shape,
             ksize=3,
             stride=1,
             padding='SAME',
             init='xavier',
             stddev=.01,
             bias=1,
             activation='relu',
             whetherBn=False,
             train=True,
             weight_decay=None,
             in_layer=None,
             init_file=None,
             init_layer_keys=None,
             trainable=None,
             trans_out_shape=None,
             reuse_name='',
             reuse_flag=None,
             batch_name='',
             batch_reuse=None,
             sm_bn_trainable=True,
             dilat=1,
             ):
        # Set parameters
        if trainable is None:
            trainable = self.default_trainable
        if in_layer is None:
            in_layer = self.output
        weight_decay = weight_decay or self.global_weight_decay
        if weight_decay is None:
            weight_decay = 0.
        if self.conv2d_data_format=='NHWC':
            in_shape = in_layer.get_shape().as_list()[-1]
            conv2d_strides = [1, stride, stride, 1]
            conv2d_dilat = [1, dilat, dilat, 1]
        else:
            in_shape = in_layer.get_shape().as_list()[1]
            conv2d_strides = [1, 1, stride, stride]
            conv2d_dilat = [1, 1, dilat, dilat]
        if isinstance(ksize, int):
            ksize1 = ksize
            ksize2 = ksize
        else:
            ksize1, ksize2 = ksize
        conv_k_shape = [ksize1, ksize2, in_shape, out_shape]
        if trans_out_shape is not None:
            conv_k_shape = [ksize1, ksize2, out_shape, in_shape]

        # Get variable
        bias_init = tf.constant_initializer(bias)
        if init=='instance_resnet':
            init = 'norm'
            stddev = np.sqrt(2.0 / (ksize1*ksize2*out_shape))
            bias_init = self.initializer(
                    'uniform',
                    stddev=np.sqrt(1.0 / (ksize1*ksize2*in_shape)),
                    )
        with tf.variable_scope(reuse_name, reuse=reuse_flag):
            if init != 'from_file':
                kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                         shape=conv_k_shape,
                                         dtype=tf.float32,
                                         regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                         name='weights', trainable = trainable)
                biases = tf.get_variable(initializer=bias_init,
                                         shape=[out_shape],
                                         dtype=tf.float32,
                                         regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                         name='bias', trainable = trainable)
            else:
                init_dict = self.initializer(init,
                                             init_file=init_file,
                                             init_keys=init_layer_keys)
                kernel = tf.get_variable(initializer=init_dict['weight'],
                                         dtype=tf.float32,
                                         regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                         name='weights', trainable = trainable)
                biases = tf.get_variable(initializer=init_dict['bias'],
                                         dtype=tf.float32,
                                         regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                         name='bias', trainable = trainable)

        if trans_out_shape is None:
            conv = tf.nn.conv2d(
                    in_layer, kernel,
                    strides=conv2d_strides,
                    dilations=conv2d_dilat,
                    padding=padding,
                    data_format=self.conv2d_data_format)
        else:
            conv = tf.nn.conv2d_transpose(
                    in_layer, kernel, 
                    output_shape=trans_out_shape,
                    strides=conv2d_strides,
                    padding=padding,
                    data_format=self.conv2d_data_format)
        if self.conv2d_with_bias:
            self.output = tf.nn.bias_add(
                    conv, biases, 
                    name='conv', data_format=self.conv2d_data_format)
        else:
            self.output = conv

        if whetherBn:
            with tf.variable_scope(reuse_name+batch_name, reuse=batch_reuse):
                self.output = self.tpu_batchnorm(train, sm_bn_trainable=sm_bn_trainable)
        if activation is not None:
            self.output = self.activation(kind=activation)

        self.params = {'input': in_layer.name,
                       'type': 'conv',
                       'num_filters': out_shape,
                       'stride': stride,
                       'kernel_size': (ksize1, ksize2),
                       'padding': padding,
                       'init': init,
                       'stddev': stddev,
                       'bias': bias,
                       'activation': activation,
                       'weight_decay': weight_decay,
                       'trans_out_shape': trans_out_shape,
                       'seed': self.seed}
        return self.output

    @tf.contrib.framework.add_arg_scope
    def fc(self,
           out_shape,
           init='xavier',
           stddev=.01,
           bias=1,
           activation='relu',
           dropout=.5,
           in_layer=None,
           init_file=None,
           init_layer_keys=None,
           trainable = None,
           weight_decay=None,
           ):

        if trainable is None:
            trainable = self.default_trainable

        weight_decay = weight_decay or self.global_weight_decay
        if weight_decay is None:
            weight_decay = 0.

        if in_layer is None:
            in_layer = self.output
        resh = tf.reshape(in_layer,
                          [in_layer.get_shape().as_list()[0], -1],
                          name='reshape')
        in_shape = resh.get_shape().as_list()[-1]

        bias_init = tf.constant_initializer(bias)
        if init=='instance_resnet':
            init = 'uniform'
            stddev = np.sqrt(1.0 / in_shape)
            bias_init = self.initializer(
                    'uniform',
                    stddev=stddev,
                    )
        if init != 'from_file':
            kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                     shape=[in_shape, out_shape],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='weights', trainable = trainable)
            biases = tf.get_variable(initializer=bias_init,
                                     shape=[out_shape],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='bias', trainable = trainable)
        else:
            init_dict = self.initializer(init,
                                         init_file=init_file,
                                         init_keys=init_layer_keys)
            kernel = tf.get_variable(initializer=init_dict['weight'],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='weights', trainable = trainable)
            biases = tf.get_variable(initializer=init_dict['bias'],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='bias', trainable = trainable)

        fcm = tf.matmul(resh, kernel)
        self.output = tf.nn.bias_add(fcm, biases, name='fc')
        if activation is not None:
            self.activation(kind=activation)
        if dropout is not None:
            self.dropout(dropout=dropout)

        self.params = {'input': in_layer.name,
                       'type': 'fc',
                       'num_filters': out_shape,
                       'init': init,
                       'bias': bias,
                       'stddev': stddev,
                       'activation': activation,
                       'dropout': dropout,
                       'weight_decay': weight_decay,
                       'seed': self.seed}
        return self.output

    @tf.contrib.framework.add_arg_scope
    def pool(self,
             ksize=3,
             stride=2,
             padding='SAME',
             pfunc='maxpool',
             in_layer=None):
        # Set parameters
        if in_layer is None:
            in_layer = self.output
        if isinstance(ksize, int):
            ksize1 = ksize
            ksize2 = ksize
        else:
            ksize1, ksize2 = ksize
        if isinstance(stride, int):
            stride1 = stride
            stride2 = stride
        else:
            stride1, stride2 = stride
        if self.conv2d_data_format=='NHWC':
            ksizes = [1, ksize1, ksize2, 1]
            strides = [1, stride1, stride2, 1]
        else:
            ksizes = [1, 1, ksize1, ksize2]
            strides = [1, 1, stride1, stride2]
        # Do the pooling
        pool_type = pfunc
        if pfunc=='maxpool':
            pool_func = tf.nn.max_pool
        else:
            pool_func = tf.nn.avg_pool
        self.output = pool_func(
                in_layer,
                ksize=ksizes,
                strides=strides,
                padding=padding,
                name='pool',
                data_format=self.conv2d_data_format)
        # Set params, return the value
        self.params = {
                'input':in_layer.name,
                'type':pool_type,
                'kernel_size': (ksize1, ksize2),
                'stride': stride,
                'padding': padding}
        return self.output

    def activation(self, kind='relu', in_layer=None):
        if in_layer is None:
            in_layer = self.output
        if kind == 'relu':
            out = tf.nn.relu(in_layer, name='relu')
        else:
            raise ValueError("Activation '{}' not defined".format(kind))
        self.output = out
        return out

    def dropout(self, dropout=.5, in_layer=None, **kwargs):
        if in_layer is None:
            in_layer = self.output
        self.output = tf.nn.dropout(in_layer, dropout, seed=self.seed, name='dropout', **kwargs)
        return self.output

    def softmax(self, in_layer=None):
        if in_layer is None:
            in_layer = self.output
        self.output = tf.nn.softmax(in_layer, name='softmax')
        self.params = {'input': in_layer.name,
                       'type': 'softmax'}
        return self.output


class NoramlNetfromConv(ConvNet):
    def __init__(self, seed=None, **kwargs):
        super(NoramlNetfromConv, self).__init__(seed=seed, **kwargs)

    @tf.contrib.framework.add_arg_scope
    def spa_disen_fc(
            self,
            out_shape,
            in_layer=None,
            weight_decay=None,
            weight_decay_type='l2',
            bias=0,
            activation=None,
            dropout=None,
            init='xavier',
            init_file=None,
            init_layer_keys=None,
            trainable=None,
            stddev=.01,
            in_conv_form=False,
            **kwargs):
        if in_layer is None:
            in_layer = self.output

        if trainable is None:
            trainable = self.default_trainable

        weight_decay = weight_decay or self.global_weight_decay
        if weight_decay is None:
            weight_decay = 0.
        weight_decay_func = tf.contrib.layers.l2_regularizer
        if weight_decay_type == 'l1':
            weight_decay_func = tf.contrib.layers.l1_regularizer

        curr_shape = in_layer.get_shape().as_list()
        if not in_conv_form:
            resh = tf.reshape(in_layer,
                              [curr_shape[0], -1],
                              name='reshape')
        else:
            resh = in_layer

        # Needs to be outputs from convolution
        assert len(curr_shape) == 4, 'Need to be output from convolution!'
        spa_shape_x = curr_shape[1]
        spa_shape_y = curr_shape[2]
        cha_shape = curr_shape[3]

        if init != 'from_file':
            spa_kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                         shape=[spa_shape_x, spa_shape_y, 1, out_shape],
                                         dtype=tf.float32,
                                         regularizer=weight_decay_func(weight_decay),
                                         name='spa_weights', trainable=trainable)
            cha_kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                         shape=[1, 1, cha_shape, out_shape],
                                         dtype=tf.float32,
                                         regularizer=weight_decay_func(weight_decay),
                                         name='cha_weights', trainable=trainable)
            kernel = spa_kernel * cha_kernel
            biases = tf.get_variable(initializer=tf.constant_initializer(bias),
                                     shape=[out_shape],
                                     dtype=tf.float32,
                                     regularizer=weight_decay_func(weight_decay),
                                     name='bias', trainable=trainable)
        else:
            init_dict = self.initializer(init,
                                         init_file=init_file,
                                         init_keys=init_layer_keys)
            spa_kernel = tf.get_variable(initializer=init_dict['weight'],
                                         dtype=tf.float32,
                                         regularizer=weight_decay_func(weight_decay),
                                         name='spa_weights', trainable=trainable)
            cha_kernel = tf.get_variable(initializer=init_dict['weight'],
                                         dtype=tf.float32,
                                         regularizer=weight_decay_func(weight_decay),
                                         name='cha_weights', trainable=trainable)
            kernel = spa_kernel * cha_kernel
            biases = tf.get_variable(initializer=init_dict['bias'],
                                     dtype=tf.float32,
                                     regularizer=weight_decay_func(weight_decay),
                                     name='bias', trainable=trainable)
        if not in_conv_form:
            kernel = tf.reshape(kernel, [-1, out_shape], name='ker_reshape')

            fcm = tf.matmul(resh, kernel)
        else:
            fcm = tf.nn.conv2d(resh, kernel,
                               strides=[1, 1, 1, 1], padding='VALID',
                               data_format=self.conv2d_data_format)

        self.output = tf.nn.bias_add(fcm, biases, name='spa_disen_fc')
        if activation is not None:
            self.activation(kind=activation)
        if dropout is not None:
            self.dropout(dropout=dropout)

        self.params = {'input': in_layer.name,
                       'type': 'spa_disen_fc',
                       'num_filters': out_shape,
                       'init': init,
                       'bias': bias,
                       'stddev': stddev,
                       'activation': activation,
                       'dropout': dropout,
                       'weight_decay': weight_decay,
                       'seed': self.seed}
        return self.output

    def random_gather_channel(
            self,
            in_layer=None,
            random_sample_ratio=0.5,
            random_sample_seed=0,):
        if in_layer is None:
            in_layer = self.output
        in_shape = in_layer.get_shape().as_list()[-1]

        random_sample = int(random_sample_ratio * in_shape)
        np.random.seed(random_sample_seed)
        rand_indx = np.random.choice(in_shape, random_sample, replace=False)
        rand_indx.sort()

        self.output = tf.gather(in_layer, indices=rand_indx, axis=-1)

        self.params = {'input': in_layer.name,
                       'type': 'random_gather_channel',
                       'random_sample_ratio': random_sample_ratio,
                       'random_sample_seed': random_sample_seed,
                       'seed': self.seed}
        return self.output

    def random_gather(
            self,
            in_layer=None,
            random_sample=1500,
            random_sample_seed=0,):
        if in_layer is None:
            in_layer = self.output
        resh = tf.reshape(in_layer,
                          [in_layer.get_shape().as_list()[0], -1],
                          name='reshape')
        in_shape = resh.get_shape().as_list()[-1]

        np.random.seed(random_sample_seed)
        if random_sample < in_shape:
            rand_indx = np.random.choice(in_shape, random_sample, replace=False)
            rand_indx.sort()
        else:
            rand_indx = range(in_shape)

        self.output = tf.gather(resh, indices=rand_indx, axis=1)

        self.params = {'input': in_layer.name,
                       'type': 'random_gather',
                       'random_sample': random_sample,
                       'random_sample_seed': random_sample_seed,
                       'seed': self.seed}
        return self.output

    @tf.contrib.framework.add_arg_scope
    def fc_with_mask(self,
                     out_shape,
                     init='xavier',
                     stddev=.01,
                     bias=1,
                     activation='relu',
                     dropout=.5,
                     in_layer=None,
                     init_file=None,
                     init_layer_keys=None,
                     trainable=None,
                     weight_decay=None,
                     random_sample=1500,
                     random_sample_seed=0,
                     ):

        if trainable is None:
            trainable = self.default_trainable

        weight_decay = weight_decay or self.global_weight_decay
        if weight_decay is None:
            weight_decay = 0.

        if in_layer is None:
            in_layer = self.output
        resh = tf.reshape(in_layer,
                          [in_layer.get_shape().as_list()[0], -1],
                          name='reshape')
        in_shape = resh.get_shape().as_list()[-1]

        np.random.seed(random_sample_seed)
        rand_indx = np.random.choice(in_shape, random_sample, replace=False)
        rand_indx.sort()

        resh = tf.gather(resh, indices=rand_indx, axis=1)
        in_shape = resh.get_shape().as_list()[-1]

        if init != 'from_file':
            kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                     shape=[in_shape, out_shape],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='weights', trainable=trainable)
            biases = tf.get_variable(initializer=tf.constant_initializer(bias),
                                     shape=[out_shape],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='bias', trainable=trainable)
        else:
            init_dict = self.initializer(init,
                                         init_file=init_file,
                                         init_keys=init_layer_keys)
            kernel = tf.get_variable(initializer=init_dict['weight'],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='weights', trainable=trainable)
            biases = tf.get_variable(initializer=init_dict['bias'],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='bias', trainable=trainable)

        fcm = tf.matmul(resh, kernel)
        self.output = tf.nn.bias_add(fcm, biases, name='fc')
        if activation is not None:
            self.activation(kind=activation)
        if dropout is not None:
            self.dropout(dropout=dropout)

        self.params = {'input': in_layer.name,
                       'type': 'fc',
                       'num_filters': out_shape,
                       'init': init,
                       'bias': bias,
                       'stddev': stddev,
                       'activation': activation,
                       'dropout': dropout,
                       'weight_decay': weight_decay,
                       'seed': self.seed}
        return self.output

    def get_trans_out_shape(
            self,
            orig_shape,
            upsample,
            number_filter):
        trans_out_shape = orig_shape 
        if self.conv2d_data_format=='NHWC':
            trans_out_shape[1] = upsample * trans_out_shape[1]
            trans_out_shape[2] = upsample * trans_out_shape[2]
            trans_out_shape[3] = number_filter
        else:
            trans_out_shape[1] = number_filter
            trans_out_shape[2] = upsample * trans_out_shape[2]
            trans_out_shape[3] = upsample * trans_out_shape[3]
        return trans_out_shape

    @tf.contrib.framework.add_arg_scope
    def resblock(self, conv_settings,
                 in_layer=None,
                 weight_decay=None,
                 bias=0,
                 padding='SAME',
                 init='xavier',
                 stddev=.01,
                 train=True,
                 reuse_flag=None,
                 batch_name='',
                 batch_reuse=None,
                 sm_trainable=None,  # sm_add
                 sm_bn_trainable=True,
                 combine_fewshot=0,
                 ):

        if in_layer is None:
            in_layer = self.output
        else:
            self.output = in_layer
        first_conv = True
        shortcut = self.output
        if self.conv2d_data_format=='NHWC':
            filter_in = self.output.get_shape().as_list()[3]
        else:
            filter_in = self.output.get_shape().as_list()[1]
        stride_all = 1
        layer_num = 0
        upsample_all = 1

        conv_kwargs = {
                'padding': padding,
                'init': init,
                'stddev': stddev,
                'bias': bias,
                'train': train,
                'activation': None,
                'weight_decay': weight_decay,
                'reuse_flag': reuse_flag,
                'batch_name': batch_name,
                'batch_reuse': batch_reuse,
                'trainable': sm_trainable,
                'sm_bn_trainable': sm_bn_trainable,
                }

        for each_setting in conv_settings:
            filter_size = each_setting['filter_size']
            stride_size = each_setting['stride']
            number_filter = each_setting['num_filters']
            whether_bn = each_setting.get('bn', 0) == 1
            upsample = each_setting.get('upsample', None)
            dilat = each_setting.get('dilat', 1)
            if combine_fewshot == 1:
                dilat = 1

            if not first_conv:
                self.output = self.activation(kind='relu')
            else:
                first_conv = False

            trans_out_shape = None
            if upsample is not None:
                trans_out_shape = self.get_trans_out_shape(
                        self.output.get_shape().as_list(),
                        upsample,
                        number_filter)
                upsample_all = upsample_all * upsample

            curr_reuse_name = 'conv_%i' % layer_num
            self.output = self.conv(
                    number_filter, filter_size, stride_size,
                    whetherBn=whether_bn,
                    reuse_name=curr_reuse_name,
                    trans_out_shape=trans_out_shape,
                    dilat=dilat,
                    **conv_kwargs
                    )

            stride_all = stride_all * stride_size
            layer_num = layer_num + 1

        if self.conv2d_data_format=='NHWC':
            filter_out = self.output.get_shape().as_list()[3]
        else:
            filter_out = self.output.get_shape().as_list()[1]
        curr_output = self.output
        if (filter_out!=filter_in) or (stride_all!=1) or (upsample_all!=1):
            trans_out_shape = None
            if upsample_all!=1:
                trans_out_shape = self.get_trans_out_shape(
                        shortcut.get_shape().as_list(),
                        upsample_all,
                        filter_out,)

            shortcut = self.conv(
                    filter_out, 1, stride=stride_all,
                    whetherBn=whether_bn,
                    in_layer=shortcut, 
                    reuse_name='transfer',
                    trans_out_shape=trans_out_shape,
                    dilat=dilat,
                    **conv_kwargs
                    )
        self.output = self.activation(
                kind='relu', 
                in_layer=curr_output + shortcut)

        self.params = {
                'input': in_layer.name,
                'type': 'ResBlock',
                'settings': conv_settings,
                'padding': padding,
                'init': init,
                'stddev': stddev,
                'bias': bias,
                'weight_decay': weight_decay,}
        return self.output

    # sm_add
    @tf.contrib.framework.add_arg_scope
    def upprojection(self, up_settings,
                     in_layer=None,
                     weight_decay=None,
                     bias=0,
                     init='xavier',
                     stddev=.01,
                     train=True,
                     reuse_flag=None,
                     batch_name='',
                     batch_reuse=None,
                     ):
        if in_layer is None:
            in_layer = self.output
        else:
            self.output = in_layer

        shortcut = self.output
        filter_size = up_settings['filter_size']
        number_filter = up_settings['num_filters']
        whether_bn = up_settings.get('bn', 0) == 1
        # Branch 1
        branch1_inter = self.unpool_as_conv(out_shape=number_filter,
                                            bias=bias,
                                            init=init,
                                            stddev=stddev,
                                            train=train,
                                            weight_decay=weight_decay,
                                            reuse_flag=reuse_flag,
                                            batch_name=batch_name,
                                            batch_reuse=batch_reuse,
                                            order=0,
                                            )
        layer_number = 0
        curr_reuse_name = 'proj_%i' % layer_number
        branch1_output = self.conv(in_layer=branch1_inter,
                                   out_shape=number_filter,
                                   ksize=filter_size,
                                   bias=bias,
                                   init=init,
                                   stddev=stddev,
                                   train=train,
                                   weight_decay=weight_decay,
                                   reuse_name=curr_reuse_name,
                                   reuse_flag=reuse_flag,
                                   batch_name=batch_name,
                                   batch_reuse=batch_reuse,
                                   whetherBn=whether_bn,
                                   )
        # Branch 2
        branch2_output = self.unpool_as_conv(in_layer=shortcut,
                                             out_shape=number_filter,
                                             bias=bias,
                                             init=init,
                                             stddev=stddev,
                                             train=train,
                                             weight_decay=weight_decay,
                                             reuse_flag=reuse_flag,
                                             batch_name=batch_name,
                                             batch_reuse=batch_reuse,
                                             order=1,
                                             )
        add_output = tf.add_n([branch1_output, branch2_output], name="up_projection")

        self.output = self.activation(kind='relu', in_layer=add_output)

        return self.output

    def reshape(self, new_size, in_layer=None):
        if in_layer is None:
            in_layer = self.output

        size_l = [in_layer.get_shape().as_list()[0]]
        size_l.extend(new_size)
        self.output = tf.reshape(in_layer, size_l)
        return self.output

    # sm_add
    def unpool_as_conv(self,
                       in_layer=None, stride=1,
                       out_shape=0, BN=True,
                       weight_decay=None,
                       bias=0,
                       init='xavier',
                       stddev=.01,
                       reuse_flag=None,
                       batch_name='',
                       batch_reuse=None,
                       train=True,
                       order=0,
                       ):
        if in_layer is None:
            in_layer = self.output
        else:
            self.output = in_layer
        current_reuse_name = 'upconv_a_%i' % order
        outputA = self.conv(ksize=[3, 3], in_layer=in_layer,
                            out_shape=out_shape, stride=stride,
                            init=init, stddev=stddev, bias=bias,
                            train=train, weight_decay=weight_decay,
                            reuse_name=current_reuse_name,
                            reuse_flag=reuse_flag,
                            batch_name=batch_name,
                            batch_reuse=batch_reuse,
                            )
        current_reuse_name = 'upconv_b_%i' % order
        tmp_layer = tf.pad(in_layer, [[0, 0], [1, 0], [1, 1], [0, 0]], "CONSTANT")
        outputB = self.conv(ksize=[2, 3], in_layer=tmp_layer,
                            out_shape=out_shape, stride=stride,
                            padding="VALID", init=init,
                            stddev=stddev, bias=bias,
                            train=train, weight_decay=weight_decay,
                            reuse_name=current_reuse_name,
                            reuse_flag=reuse_flag,
                            batch_name=batch_name,
                            batch_reuse=batch_reuse,
                            )
        current_reuse_name = 'upconv_c_%i' % order
        tmp_layer = tf.pad(in_layer, [[0, 0], [1, 1], [1, 0], [0, 0]], "CONSTANT")
        outputC = self.conv(ksize=[3, 2], in_layer=tmp_layer,
                            out_shape=out_shape, stride=stride,
                            padding="VALID", init=init,
                            stddev=stddev, bias=bias,
                            train=train, weight_decay=weight_decay,
                            reuse_name=current_reuse_name,
                            reuse_flag=reuse_flag,
                            batch_name=batch_name,
                            batch_reuse=batch_reuse,
                            )
        current_reuse_name = 'upconv_d_%i' % order
        tmp_layer = tf.pad(in_layer, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
        outputD = self.conv(ksize=[2, 2], in_layer=tmp_layer,
                            out_shape=out_shape, stride=stride,
                            padding="VALID", init=init,
                            stddev=stddev, bias=bias,
                            train=train, weight_decay=weight_decay,
                            reuse_name=current_reuse_name,
                            reuse_flag=reuse_flag,
                            batch_name=batch_name,
                            batch_reuse=batch_reuse,
                            )

        def get_incoming_shape(incoming):
            # Returns the incoming data shape
            if isinstance(incoming, tf.Tensor):
                return incoming.get_shape().as_list()
            elif type(incoming) in [np.array, list, tuple]:
                return np.shape(incoming)
            else:
                raise Exception("Invalid incoming layer.")

        def interleave(tensors, axis):
            old_shape = get_incoming_shape(tensors[0])[1:]
            new_shape = [-1] + old_shape
            new_shape[axis] *= len(tensors)
            return tf.reshape(tf.stack(tensors, axis + 1), new_shape)

        left = interleave([outputA, outputB], axis=1)
        right = interleave([outputC, outputD], axis=1)
        self.output = interleave([left, right], axis=2)

        current_reuse_name = 'bn_%i' % order
        if BN:
            with tf.variable_scope(current_reuse_name, reuse=reuse_flag):
                self.output = self.tpu_batchnorm(train)
        self.output = self.activation(kind='relu')
        return self.output

    def resize_images(self, new_size, in_layer=None):
        if in_layer is None:
            in_layer = self.output
        self.output = tf.image.resize_images(in_layer, [new_size, new_size])
        return self.output

    def add_bypass_adding(self, bypass_layer, in_layer=None):
        if in_layer is None:
            in_layer = self.output

        bypass_shape = bypass_layer.get_shape().as_list()
        if len(bypass_shape) == 4:
            ds = in_layer.get_shape().as_list()[1]
            if not ds is None:
                if bypass_shape[1] != ds:
                    bypass_layer = tf.image.resize_images(bypass_layer, [ds, ds])
            else:
                ds = tf.shape(in_layer)[1]
                bypass_layer = tf.image.resize_bilinear(bypass_layer, [ds, ds], align_corners=False)

        self.output = in_layer + bypass_layer

        self.params = {'input': in_layer.name,
                       'bypass_layer': bypass_layer.name,
                       'type': 'bypass_adding'}

        return self.output

    def add_bypass(self, bypass_layer, in_layer=None):
        if in_layer is None:
            in_layer = self.output
        bypass_shape = bypass_layer.get_shape().as_list()
        if len(bypass_shape) == 4:
            ds = in_layer.get_shape().as_list()[1]
            if not ds is None:
                ds2 = in_layer.get_shape().as_list()[2]  # sm_modify
                if bypass_shape[1] != ds or bypass_shape[2] != ds2:  # sm_modify
                    bypass_layer = tf.image.resize_images(bypass_layer, [ds, ds2])  # sm_modify
            else:
                ds = tf.shape(in_layer)[1]
                bypass_layer = tf.image.resize_bilinear(bypass_layer, [ds, ds], align_corners=False)

            self.output = tf.concat([in_layer, bypass_layer], 3)
        else:
            self.output = tf.concat([in_layer, bypass_layer], len(bypass_shape) - 1)

        self.params = {'input': in_layer.name,
                       'bypass_layer': bypass_layer.name,
                       'type': 'bypass'}

        return self.output

    def resize_images_scale(self, scale, in_layer=None):
        if in_layer is None:
            in_layer = self.output

        curr_shape = in_layer.get_shape().as_list()
        if (curr_shape[1] is None) or (curr_shape[2] is None):
            im_h = tf.shape(in_layer)[1]
            im_w = tf.shape(in_layer)[2]
        else:
            im_h = curr_shape[1]
            im_w = curr_shape[2]
        self.output = tf.image.resize_bilinear(
                in_layer, [im_h * scale, im_w * scale], align_corners=False)
        return self.output

    def ae_head(self, dimension, in_layer=None, **fc_kwargs):
        if in_layer is None:
            in_layer = self.output
        curr_shape = in_layer.get_shape().as_list()
        flatten_dim = np.prod(curr_shape[1:])
        in_layer = tf.reshape(in_layer, [curr_shape[0], flatten_dim])

        with tf.variable_scope('compress_fc'):
            self.output = self.fc(
                    out_shape=dimension, in_layer=in_layer, **fc_kwargs)
        tf.add_to_collection('AE_embedding', self.output)

        with tf.variable_scope('decompress_fc'):
            self.output = self.fc(
                    out_shape=flatten_dim, **fc_kwargs)
        self.output = tf.reshape(self.output, curr_shape)
        return self.output
