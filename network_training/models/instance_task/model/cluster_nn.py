from __future__ import division, print_function

import numpy as np
import tensorflow as tf

def cluster_nn(nns):
    n = len(nns)
    parent = range(n)
    def get_parent(x):
        if parent[parent[x]] != parent[x]:
            parent[x] = get_parent(parent[x])
        return parent[x]

    for a, b in enumerate(nns):
        parent[get_parent(a)] = get_parent(b)

    cluster_sizes = {}
    remap_cluster_ids = {}

    for i in range(n):
        p = get_parent(i)
        if p not in cluster_sizes:
            remap_cluster_ids[p] = len(cluster_sizes)
            cluster_sizes[p] = 0
        cluster_sizes[p] += 1

    cluster_labels = np.array(
        [remap_cluster_ids[get_parent(i)] for i in range(n)]
    )
    return cluster_labels, cluster_sizes


class NNClustering(object):
    def __init__(self, steps_per_epoch, nearest_neighbors, cluster_labels):
        self.steps_per_epoch = steps_per_epoch
        self.nn = nearest_neighbors
        self.clusters = cluster_labels
        self.new_cluster_feed = tf.placeholder(
            tf.int64, shape=self.clusters.get_shape().as_list())
        self.update_clusters_op = tf.assign(self.clusters, self.new_cluster_feed)

    def recompute_clusters(self, sess):
        nns = sess.run(self.nn)
        new_clust_labels, cluster_sizes = cluster_nn(nns)
        print("New clustering has", len(cluster_sizes), "clusters")
        print("Top 20 sizes:", sorted(cluster_sizes.values(), reverse=True)[:20])
        sess.run(self.update_clusters_op, feed_dict={
            self.new_cluster_feed: new_clust_labels
        })
        print("Cluster labels are now:", sess.run(self.clusters))
