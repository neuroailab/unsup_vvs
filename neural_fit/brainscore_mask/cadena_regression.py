import numpy as np
import pdb
from tqdm import tqdm
import logging
from faster_mask_regression import \
        get_iter_from_one_input, get_iter_from_two_inputs
import sys
import os
from brainscore.utils import fullname


class CadenaRegression(object):
    def __init__(
            self,
            smooth_reg_weight=0.1, 
            sparse_reg_weight=0.01, 
            group_sparsity_weight=0.01,
            output_nonlin_smooth_weight=-1,
            b_norm=True,
            batch_size=256,
            _decay_rate=[300, 375, 400],
            max_epochs=440,
            init_lr=1e-4,
            *args, **kwargs):
        self.smooth_reg_weight = smooth_reg_weight
        self.sparse_reg_weight = sparse_reg_weight
        self.group_sparsity_weight = group_sparsity_weight
        self.b_norm = b_norm
        self._batch_size = batch_size
        self._decay_rate = _decay_rate
        self._max_epochs = max_epochs
        self._lr = init_lr
        assert output_nonlin_smooth_weight == -1, "Adding non lin not supported"

        self._graph = None
        self._lr_ph = None
        self._opt = None
        self._logger = logging.getLogger(fullname(self))

    def fit(self, X, Y):
        assert not np.isnan(X).any()
        self.setup()
        X = self.reindex(X)

        assert X.ndim == 4, 'Input matrix rank should be 4.'
        with self._graph.as_default():
            self._init_mapper(X, Y)
            lr = self._lr
            for epoch in tqdm(range(self._max_epochs), desc='mask epochs'):
                for _ in range(0, len(X), self._batch_size):
                    feed_dict = {self._lr_ph: lr}
                    _ = self._sess.run(
                            [self.train_op],
                            feed_dict=feed_dict)
                if (len(self._decay_rate)>0) \
                        and (epoch % self._decay_rate[0] == 0 and epoch != 0):
                    lr /= 3.
                    self._decay_rate.remove(self._decay_rate[0])

    def predict(self, X):
        self.is_training = False
        import tensorflow as tf
        assert not np.isnan(X).any()
        X = self.reindex(X)
        with self._graph.as_default():
            self._input = get_iter_from_one_input(X, self._batch_size).get_next()
            self._make_map()

            preds = []
            for _ in range(0, len(X), self._batch_size):
                preds.append(
                        np.squeeze(self._sess.run([self._predictions])))
            final_preds = np.concatenate(preds, axis=0)
            return final_preds

    def setup(self):
        import tensorflow as tf
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._lr_ph = tf.placeholder(dtype=tf.float32)
            self._opt = tf.train.AdamOptimizer(learning_rate=self._lr_ph)

    def reindex(self, X):
        channel_names = ['channel', 'channel_x', 'channel_y']
        assert all(hasattr(X, coord) for coord in channel_names)
        shapes = [len(set(X[channel].values)) for channel in channel_names]
        X = np.reshape(X.values, [X.shape[0]] + shapes)
        X = np.transpose(X, axes=[0, 2, 3, 1])
        return X

    def _make_map(self):
        import tensorflow as tf
        sys.path.append(os.path.expanduser('~/Cadena2019PlosCB'))
        from cnn_sys_ident import vggsysid
        with self._graph.as_default():
            with tf.variable_scope('mapping', reuse=tf.AUTO_REUSE):
                if self.b_norm:
                    self._input = tf.layers.batch_normalization(
                            self._input, training=self.is_training,
                            momentum=0.9, epsilon=1e-4, fused =True)
                self._input = tf.nn.relu(self._input)
                self._predictions = vggsysid.readout(
                        self._input, 
                        self.out_shape, 
                        self.smooth_reg_weight, 
                        self.sparse_reg_weight, 
                        self.group_sparsity_weight)
                self._predictions = tf.nn.elu(self._predictions - 1.0) + 1.0

    def _make_loss(self):
        import tensorflow as tf
        prediction = self._predictions
        response = self._target
        realresp = tf.math.logical_not(tf.math.is_nan(response))
        prediction = tf.boolean_mask(prediction, realresp)
        response = tf.boolean_mask(response, realresp)
        self._pred_loss = tf.reduce_mean(
                    prediction - response * tf.log(prediction + 1e-9))
        self._reg_loss = tf.add_n(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.total_loss = self._pred_loss + self._reg_loss
        self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print_op = tf.print(
                'L2 error: ', self._pred_loss, 
                'Total loss: ', self.total_loss
                )
        print_op = tf.no_op()
        update_ops.append(print_op)
        with tf.control_dependencies(update_ops):
            self.train_op = self._opt.minimize(
                    self.total_loss, var_list=self.tvars,
                global_step=tf.train.get_or_create_global_step())

    def _init_mapper(self, X, Y):
        self.is_training = True
        import tensorflow as tf
        assert len(Y.shape) == 2
        self.out_shape = Y.shape[1]
        with self._graph.as_default():
            # Make input
            self._input, self._target = get_iter_from_two_inputs(X, Y, self._batch_size).get_next()
            # Build the model graph
            self._make_map()
            self._make_loss()

            # initialize graph
            self._logger.debug('Initializing mapper')
            init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            gpu_options = tf.GPUOptions(allow_growth=True)
            self._sess = tf.Session(
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        gpu_options=gpu_options,
                        ))
            self._sess.run(init_op)
