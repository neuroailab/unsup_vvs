from __future__ import division, print_function
import numpy as np
import tensorflow as tf
import os

from sklearn import svm
import cPickle

import model.preprocessing as prep
from utils import DATA_LEN_IMAGENET_FULL
import tfutils_reader
from tfutils_reader import TfutilsReader


def augment_img(img_str, num_reps):
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session(graph=g,
                          config=tf.ConfigProto(device_count={'GPU':0}))
        center_crop = sess.run(prep.resnet_validate(img_str))
        aug = prep.resnet_train(img_str)
        augs = []
        for i in xrange(num_reps):
            augs.append(sess.run(aug))
        sess.close()
        return augs, center_crop


def calculate_embeddings(sess, model, img_list, print_progress=True):
    start = 0
    batch_size = 128
    vecs = []
    while start < len(img_list):
        if print_progress:
            print('Evaluating embeddings', start, 'to', start + batch_size)
        batch = sess.run(model['embedding'],
            feed_dict={
                model['input_handle']: img_list[start:start+batch_size]
            })
        vecs.append(batch)
        start += batch_size
    vecs = np.concatenate(vecs)
    return vecs


class MemoryBank(object):
    @staticmethod
    def load(tfu_reader, step_num, gpu,
             shape=(DATA_LEN_IMAGENET_FULL, 128)):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        tf.reset_default_graph() # TODO: do this loading in a non-global way
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        model_data = tfutils_reader.load_model(
            sess, tfu_reader, step_num=step_num, load_mem_bank=True,
            mem_bank_shape=shape
        )
        mb = sess.run(model_data['memory_bank'])
        labels = sess.run(model_data['labels'])
        ret = MemoryBank(mb, labels)
        sess.close()
        return ret

    def __init__(self, memory_vecs, labels, dim=128):
        self.dim = dim

        self.num_classes = 1000
        self.n = len(labels)
        self.memory_vecs = memory_vecs
        self.labels = labels

        self.class_sizes = np.zeros(self.num_classes)
        self.class_means = np.zeros((self.num_classes, self.dim))
        self.class_to_idxs = [[] for _ in xrange(self.num_classes)]

        for i in xrange(self.n):
            c = self.labels[i]
            self.class_to_idxs[c].append(i)
            self.class_sizes[c] += 1
            self.class_means[c] += memory_vecs[i]
        for c in xrange(self.num_classes):
            self.class_means[c] /= self.class_sizes[c]

    def get_k_closest(self, vec, k):
        assert k > 0
        all_dps = np.dot(self.memory_vecs, vec)
        sorted_idxs = np.argsort(all_dps)
        sorted_idxs = sorted_idxs[-k:]
        dps = all_dps[sorted_idxs]
        return list(reversed(zip(dps, sorted_idxs)))

    def memories_by_class(self, class_id):
        idxs = self.class_to_idxs[class_id]
        return self.memory_vecs[idxs]

    def classify(self, batch_vecs): # TODO: top k
        N = len(batch_vecs)
        ret = []

        idx = 0
        batch_size = 16
        while idx < N:
            dps = np.dot(batch_vecs[idx:idx + batch_size], self.memory_vecs.T)
            for i, row in enumerate(dps):
                top = row.argsort()[-1]
                ret.append(self.labels[top])
            idx += batch_size
        return ret


# if __name__ == '__main__':
#     tr = TfutilsReader('instance_task', 'control', 'full', port=27009,
#                        cache_dir='/mnt/fs1/azhai/checkpoints')
#     mb = MemoryBank.load(tr, step_num=20000, gpu=2)
