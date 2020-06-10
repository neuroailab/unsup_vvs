from brainscore.metrics.regression import MaskRegression
import numpy as np
import pdb
from nf_utils import compute_corr
from tqdm import tqdm
import os


def get_iter_from_one_input(X, batch_size):
    import tensorflow as tf
    def _dataset_gen():
        for _idx in range(len(X)):
            yield X[_idx]
    zip_dataset = tf.data.Dataset.from_generator(
            _dataset_gen, 
            output_types=(tf.float32),
            output_shapes=(X.shape[1:]))
    zip_dataset = zip_dataset.cache()
    zip_dataset = zip_dataset.batch(batch_size)
    zip_dataset = zip_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    zip_iter = zip_dataset.make_one_shot_iterator()
    return zip_iter


def get_iter_from_two_inputs(X, Y, batch_size):
    import tensorflow as tf
    def _dataset_gen():
        for _idx in range(len(X)):
            yield X[_idx], Y[_idx]
    zip_dataset = tf.data.Dataset.from_generator(
            _dataset_gen, 
            output_types=(tf.float32, tf.float32),
            output_shapes=(X.shape[1:], Y.shape[1:]))
    zip_dataset = zip_dataset.cache()
    zip_dataset = zip_dataset.shuffle(buffer_size=len(X)).repeat()
    zip_dataset = zip_dataset.batch(batch_size)
    zip_dataset = zip_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    zip_iter = zip_dataset.make_one_shot_iterator()
    return zip_iter


class FasterMaskRegression(MaskRegression):
    def __init__(
            self, 
            with_corr_loss=False,
            with_lap_loss=True,
            adam_eps=1e-08,
            normalize_Y=False,
            checkpoint_dir=None,
            *args, **kwargs):
        self.with_corr_loss = with_corr_loss
        self.with_lap_loss = with_lap_loss
        self.adam_eps = adam_eps
        self.normalize_Y = normalize_Y
        self.checkpoint_dir = checkpoint_dir
        super().__init__(*args, **kwargs)

    def _make_separable_map(self):
        """
        Makes the mapping function computational graph
        """
        import tensorflow as tf
        out_shape = self.out_shape
        with self._graph.as_default():
            with tf.variable_scope('mapping', reuse=tf.AUTO_REUSE):
                input_shape = self._input.shape
                _, spa_x_shape, spa_y_shape, depth_shape = input_shape

                s_w_shape = (spa_x_shape, spa_y_shape, 1, out_shape)
                if self._inits is not None and 's_w' in self._inits:
                    s_w = tf.Variable(initial_value=\
                                          self._inits['s_w'].reshape(s_w_shape),
                                      dtype=tf.float32)
                else:
                    s_w = tf.get_variable(
                            name='spatial_mask',
                            initializer=tf.contrib.layers.xavier_initializer(),
                            shape=s_w_shape,
                            dtype=tf.float32)

                d_w_shape = (1, 1, depth_shape, out_shape)
                if self._inits is not None and 'd_w' in self._inits:
                    d_w = tf.Variable(initial_value=\
                                          self._inits['d_w'].reshape(d_w_shape),
                                      dtype=tf.float32)
                else:
                    d_w = tf.get_variable(
                            name='depth_mask',
                            initializer=tf.contrib.layers.xavier_initializer(),
                            shape=d_w_shape,
                            dtype=tf.float32)

                if self._inits is not None and 'bias' in self._inits:
                    bias = tf.Variable(initial_value=\
                                           self._inits['bias'].reshape(out_shape),
                                       dtype=tf.float32)
                else:
                    bias = tf.get_variable(
                            name='bias',
                            initializer=tf.contrib.layers.xavier_initializer(),
                            shape=out_shape,
                            dtype=tf.float32)

                tf.add_to_collection('s_w', s_w)
                tf.add_to_collection('d_w', d_w)
                tf.add_to_collection('bias', bias)

                if self.checkpoint_dir:
                    self._saver = tf.train.Saver(tf.global_variables())
                kernel = s_w * d_w
                kernel = tf.reshape(kernel, [-1, out_shape])
                inputs = tf.layers.flatten(self._input)
                self._predictions = tf.matmul(inputs, kernel)
                self._predictions = tf.nn.bias_add(self._predictions, bias)

    def setup(self):
        import tensorflow as tf
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._lr_ph = tf.placeholder(dtype=tf.float32)
            self._opt = tf.train.AdamOptimizer(
                    learning_rate=self._lr_ph, 
                    epsilon=self.adam_eps)

    def _get_l2_loss(self, t):
        import tensorflow as tf
        t = tf.reshape(t, shape=(-1,))
        real_mask = tf.math.logical_not(tf.math.is_nan(t))
        num_reals = tf.reduce_sum(tf.cast(real_mask, tf.float32))
        t_masked = tf.boolean_mask(t, real_mask)
        return tf.nn.l2_loss(t_masked) / num_reals

    def _make_loss(self):
        """
        Makes the loss computational graph
        """
        import tensorflow as tf
        with self._graph.as_default():
            with tf.variable_scope('loss'):
                self.l2_error = self._get_l2_loss(
                        self._predictions - self._target)
                # For separable mapping
                self._s_vars = tf.get_collection('s_w')
                self._d_vars = tf.get_collection('d_w')
                self._s_vars = tf.split(
                        self._s_vars[0], 
                        num_or_size_splits=self._s_vars[0].shape[-1], axis=-1)

                # Laplacian loss
                laplace_filter = tf.constant(
                        np.array([0, -1, 0, -1, 4, -1, 0, -1, 0])\
                          .reshape((3, 3, 1, 1)),
                        dtype=tf.float32)
                laplace_loss = tf.reduce_sum(
                    [tf.norm(
                        tf.nn.conv2d(t, laplace_filter, [1, 1, 1, 1], 'SAME'))\
                     for t in self._s_vars])
                s_l2_loss = tf.reduce_sum(
                        [tf.nn.l2_loss(t) for t in self._s_vars])
                d_l2_loss = tf.reduce_sum(
                        [tf.nn.l2_loss(t) for t in self._d_vars])
                self.reg_loss = self._ls * s_l2_loss + self._ld * d_l2_loss
                if self.with_lap_loss:
                    self.reg_loss += self._ls * laplace_loss

                self.total_loss = self.l2_error + self.reg_loss
                if self.with_corr_loss:
                    self.total_loss \
                            -= compute_corr(
                                    self._predictions, 
                                    self._target)
                print_op = tf.print(
                        'L2 error: ', self.l2_error, 
                        'Reg loss: ', self.reg_loss,
                        'Avg resp: ', tf.reduce_mean(self._target),
                        'Std resp: ', tf.math.reduce_std(self._target),
                        'Avg pred: ', tf.reduce_mean(self._predictions),
                        'Std pred: ', tf.math.reduce_std(self._predictions),
                        )
                print_op = tf.no_op()

                self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                with tf.control_dependencies([print_op]):
                    self.train_op = self._opt.minimize(
                            self.total_loss, var_list=self.tvars,
                        global_step=tf.train.get_or_create_global_step())

    def fit(self, X, Y):
        """
        Fits the parameters to the data
        :param X: Source data, first dimension is examples
        :param Y: Target data, first dimension is examples
        """
        assert not np.isnan(X).any()
        self.setup()
        X = self.reindex(X)
        if self.normalize_Y:
            Y_raw_arr = np.asarray(Y)
            self._mean_Y = np.nanmean(Y_raw_arr, axis=0, keepdims=True)
            self._std_Y = np.nanstd(Y_raw_arr, axis=0, keepdims=True)
            Y = (Y - self._mean_Y) / self._std_Y

        assert X.ndim == 4, 'Input matrix rank should be 4.'
        with self._graph.as_default():
            self._init_mapper(X, Y)
            lr = self._lr
            for epoch in tqdm(range(self._max_epochs), desc='mask epochs'):
                for _ in range(0, len(X), self._batch_size):
                    feed_dict = {self._lr_ph: lr}
                    _, loss_value, reg_loss_value = self._sess.run(
                            [self.train_op, self.l2_error, self.reg_loss],
                            feed_dict=feed_dict)
                if epoch % self._log_rate == 0:
                    self._logger.debug(f'Epoch: {epoch}, Err Loss: {loss_value:.2f}, Reg Loss: {reg_loss_value:.2f}')
                if epoch % self._decay_rate == 0 and epoch != 0:
                    lr /= 10.
                if loss_value < self._tol:
                    self._logger.debug('Converged.')
                    break
            if self.checkpoint_dir:
                # Tricky way to get current split
                curr_split = 0
                while True:
                    curr_dir = os.path.join(self.checkpoint_dir, 'split_' + str(curr_split))
                    if not os.path.isdir(curr_dir):
                        os.system('mkdir -p ' + curr_dir)
                        self._saver.save(
                                self._sess, os.path.join(curr_dir, 'model.ckpt'))
                        break
                    curr_split += 1

    def predict(self, X):
        """
        Predicts the responses to the give input X
        :param X: Input data, first dimension is examples
        :return: predictions
        """
        import tensorflow as tf
        assert not np.isnan(X).any()
        X = self.reindex(X)
        with self._graph.as_default():
            self._input = get_iter_from_one_input(X, self._batch_size).get_next()
            self._make_separable_map()

            preds = []
            for _ in range(0, len(X), self._batch_size):
                preds.append(
                        np.squeeze(self._sess.run([self._predictions])))
            final_preds = np.concatenate(preds, axis=0)
            if self.normalize_Y:
                final_preds = final_preds * self._std_Y + self._mean_Y
            return final_preds

    def _init_mapper(self, X, Y):
        """
        Initializes the mapping function graph
        :param X: input data
        """
        import tensorflow as tf
        assert len(Y.shape) == 2
        self.out_shape = Y.shape[1]
        with self._graph.as_default():
            # Make input
            self._input, self._target = get_iter_from_two_inputs(X, Y, self._batch_size).get_next()
            # Build the model graph
            self._make_separable_map()
            self._make_loss()

            # initialize graph
            self._logger.debug('Initializing mapper')
            init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            #self._sess = tf.Session(
            #    config=tf.ConfigProto(gpu_options=self._gpu_options) if self._gpu_options is not None else None)
            gpu_options = tf.GPUOptions(allow_growth=True)
            self._sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=gpu_options,
                ))
            self._sess.run(init_op)
