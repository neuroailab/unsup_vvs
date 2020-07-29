from __future__ import division, print_function
import os, sys
import tensorflow as tf
import numpy as np
import argparse
import itertools
import time

def find_kmeans(points, k, max_iter=100, random_seed=0):
    def all_pts_dataset():
        return tf.train.limit_epochs(
            tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

    kmeans = tf.contrib.factorization.KMeansClustering(
        num_clusters=k, use_mini_batch=False,
        random_seed=random_seed,
    )

    last_time = time.time()
    score = centers = prev_centers = None
    for _ in range(max_iter):
        kmeans.train(all_pts_dataset)
        centers = kmeans.cluster_centers()
        if prev_centers is not None:
            delta = centers - prev_centers
            if np.linalg.norm(delta) < 1e-6:
                break
        prev_centers = centers
        score = kmeans.score(all_pts_dataset)
        print('score:', score, '[%f]' % (time.time() - last_time))
        last_time = time.time()

    cluster_indices = list(kmeans.predict_cluster_index(all_pts_dataset))
    return centers, np.array(cluster_indices), score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Perform kmeans using tensorflow.")
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--outpath', type=str, required=True)
    parser.add_argument('--embeddings_path', type=str,
                        default='/mnt/fs3/azhai/center_cropped_embeddings/ep220_embeddings.npy')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu or ''

    cc_embeddings = np.load(args.embeddings_path)
    print('Loaded embeddings', cc_embeddings.shape)

    centers, idxs, score = find_kmeans(cc_embeddings, k=1000, random_seed=args.seed)
    print(centers.shape, idxs.shape)
    np.savez(args.outpath, centers=centers, indices=idxs, score=score)
