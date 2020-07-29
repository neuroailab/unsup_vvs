import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm


def get_clstr_labels_and_index(clstr_path, indx_for_clstr):
    assert os.path.exists(clstr_path) and clstr_path.endswith('npy'), \
            "Cluster file does not exist or end with npy!"
    clstr_labels = np.load(clstr_path)
    num_of_labels = len(clstr_labels)
    if not indx_for_clstr:
        print('Will assume the labels are for the first part of images!')
        label_index = np.arange(num_of_labels)
    else:
        label_index = np.load(indx_for_clstr)
        assert len(label_index) == num_of_labels, \
                "Numbers of labels and indexes do not match!"
    return clstr_labels, label_index


class ClusteringBase(object):
    def __init__(self, memory_bank, cluster_labels):
        self.memory_bank = memory_bank
        self.cluster_labels = cluster_labels

        self.new_cluster_feed = tf.placeholder(
            tf.int64, shape=self.cluster_labels.get_shape().as_list())
        self.update_clusters_op = tf.assign(
                self.cluster_labels, self.new_cluster_feed)

    def apply_clusters(self, sess, new_clust_labels):
        sess.run(self.update_clusters_op, feed_dict={
            self.new_cluster_feed: new_clust_labels
        })


class LabelClustering(ClusteringBase):
    """
    Cluster based on known labels
    """
    def __init__(
            self, 
            clstr_path, memory_bank, cluster_labels, nearest_neighbors,
            indx_for_clstr=None):
        super(LabelClustering, self).__init__(memory_bank, cluster_labels)
        self.clstr_labels, self.label_index \
                = get_clstr_labels_and_index(clstr_path, indx_for_clstr)
        self.nearest_neighbors = nearest_neighbors

    def recompute_clusters(self, sess):
        nns = sess.run(self.nearest_neighbors)
        if np.max(nns) > len(self.label_index):
            print('Using trivial labels')
            new_clstr_labels = np.arange(len(nns))
        else:
            new_clstr_labels = self.clstr_labels[self.label_index[nns]]
        new_clstr_labels = new_clstr_labels[np.newaxis, :]
        return new_clstr_labels
