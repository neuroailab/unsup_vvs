import tensorflow as tf
import pdb
import numpy as np


def image_preprocess(x):
    x = tf.image.resize_images(x, (256, 256))
    arr = []
    for j in range(7):
        for i in range(7):
            curr_crop = x[:, i*32:i*32+64, j*32:j*32+64, :]
            curr_crop = tf.expand_dims(curr_crop, axis=1)
            arr.append(curr_crop)
    x = tf.concat(arr, axis=1)
    x = tf.reshape(x, [-1] + x.get_shape().as_list()[2:])
    return x


class CPC():
    def __init__(
            self, X, X_len, Y, Y_label, 
            n=7, code=7, k=2, code_dim=512, cell_dimension=128):
        """
        Autoregressive part from CPC
        """

        with tf.variable_scope('CPC'):
            self.X = X
            self.X_len = X_len
            self.Y = Y
            self.Y_label = Y_label
            self.batch_size = X.shape[0]


            self.n = n
            self.k = k
            self.code = code
            self.code_dim = code_dim
            self.cell_dimension = cell_dimension

            cell = tf.contrib.rnn.GRUCell(cell_dimension, name='cell')
            initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

            # Autoregressive model
            with tf.variable_scope('g_ar'):
                _, c_t = tf.nn.dynamic_rnn(
                        cell, self.X, 
                        sequence_length=self.X_len, 
                        initial_state=initial_state)
                self.c_t = c_t

            with tf.variable_scope('coding'):
                losses = []
                y = self.Y
                for i in range(k):
                    W = tf.get_variable(
                            'x_t_'+str(i+1), 
                            shape=[cell_dimension, self.code_dim])
                    y_ = tf.reshape(
                            y[:, i, :], 
                            [self.batch_size, self.code_dim])
                    self.cpc = tf.map_fn(
                            lambda x: tf.squeeze(
                                tf.matmul(tf.transpose(W), tf.expand_dims(x, -1)), 
                                axis=-1), 
                            c_t) * y_
                    nce = tf.sigmoid(tf.reduce_mean(self.cpc, -1))
                    losses.append(
                            tf.keras.losses.binary_crossentropy(
                                self.Y_label, nce))

                losses = tf.stack(losses, axis=0)

            # Loss function
            with tf.variable_scope('train'):
                self.loss = tf.reduce_mean(losses)


def build_cpc_loss(features):
    curr_shape = features.get_shape().as_list()
    features = tf.reduce_mean(features, axis=[1, 2])
    batch_size = curr_shape[0] // 49
    features = tf.reshape(
            features, 
            shape=[batch_size, 7, 7, curr_shape[-1]])
    n = 7
    K = 2
    tmp = []
    for i in range(batch_size):
        for j in range(n):
            tmp.append(features[i][j])
    X = tf.stack(tmp, 0)
    batch_size *= n

    # for random row
    nl = []
    nrr = []
    nrri = []
    for i in range(K):
        nlist = np.arange(0, n)
        nlist = nlist[nlist != (n-K+i)]
        nl.append(tf.constant(nlist))
        nrr.append([tf.random_shuffle(nl[i]) for j in range(batch_size)])
    nrri = [tf.stack([nrr[j][i][0] for j in range(K)], axis=0) for i in range(batch_size)]

    Y = []
    Y_label = np.zeros((batch_size), dtype=np.float32)
    n_p = batch_size // 2

    for i in range(batch_size):
        if i <= n_p:
            Y.append(tf.expand_dims(features[int(i/n), -K:, i%n, :], axis=0))
            Y_label[i] = 1
        else:
            Y.append(tf.expand_dims(tf.gather(features[int(i/n)], nrri[i])[:, i%n, :], axis=0))

    Y = tf.concat(Y, axis=0)
    Y_label = tf.constant(Y_label, dtype=np.float32)

    nr = tf.random_shuffle(tf.constant(list(range(batch_size)), dtype=tf.int32))

    ## cpc
    X_len = [5] * batch_size
    X_len = tf.constant(X_len, dtype=tf.int32)

    cpc = CPC(X, X_len, Y, Y_label, k=K)
    return cpc.loss
