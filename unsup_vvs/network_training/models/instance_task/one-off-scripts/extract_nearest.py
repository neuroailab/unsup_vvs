from __future__ import division, print_function
import os, sys
import tensorflow as tf
import numpy as np
import argparse
import itertools
import time

from sklearn.neighbors import NearestNeighbors

# Add parent directory to the system path so we can import from there.
parent, _ = os.path.split(os.getcwd())
if parent not in sys.path:
    sys.path.append(parent)

from tfutils_reader import TfutilsReader, load_model
from embedding_stats import MemoryBank


def load_mb_from_npy(embeddings_path):
    labels = np.load('/mnt/fs3/azhai/center_cropped_imagenet/all_labels.npy')
    memory_vecs = np.load(embeddings_path)
    return MemoryBank(memory_vecs, labels)


def get_nearest_list(mb, save_dir, epoch, start_idx=0, end_idx=None, topk=5000):
    start = time.time()

    end_idx = end_idx or mb.n
    end_idx = min(end_idx, mb.n)
    print("Loaded memory bank", time.time() - start)

    all_vecs = tf.get_variable('embeddings', shape=(128, mb.n),
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer, trainable=False)
    batch = tf.placeholder(tf.float32, shape=(None, 128))
    all_dps = tf.matmul(batch, all_vecs)
    top_dps = tf.nn.top_k(all_dps, k=topk, sorted=True)
    print("Set up computation graph", time.time() - start)

    sess = tf.Session()
    sess.run(tf.assign(all_vecs, mb.memory_vecs.T))

    i = start_idx
    batch_size = 128
    highest_dps = []
    highest_dp_indices = []
    while i < end_idx:
        print("Computing nearest neighbors index", i, time.time() - start)
        values, indices = sess.run(top_dps, feed_dict={
            batch: mb.memory_vecs[i:min(i + batch_size, end_idx)],
        })
        i += batch_size
        highest_dps.append(values)
        highest_dp_indices.append(indices)
    sess.close()

    print("Finished computations", time.time() - start)
    highest_dps = np.concatenate(highest_dps)
    highest_dp_indices = np.concatenate(highest_dp_indices)
    print('highest_dps', highest_dps.shape)
    print('highest_dp_indices', highest_dp_indices.shape)

    if start_idx == 0 and end_idx == mb.n:
        outpath = os.path.join(
            save_dir, 'ep%i_nearest_list_top%i.npz' % (epoch, topk))
    else:
        outpath = os.path.join(
            save_dir, 'ep%i_nearest_list_%i_to_%i.npz' % (epoch, start_idx, end_idx))
    print("Saving results to %s" % outpath)
    np.savez(open(outpath, 'w'), highest_dps=highest_dps,
             highest_dp_indices=highest_dp_indices)


def concatenate_files(epoch, k):
    full_data = {
        'highest_dps': [],
        'highest_dp_indices': []
    }
    for i in range(129):
        start, end = i * 10000, (i+1) * 10000
        end = min(end, 1281167)
        print('Processing range', start, 'to', end)
        nbrs = np.load('/mnt/fs3/azhai/center_cropped_embeddings/nearest_neighbors/'
                       'ep%i_nearest_list_%i_to_%i.npz' % (epoch, start, end))
        for key in full_data:
            # top k reversed
            top = nbrs[key][:,-1:-k-1:-1]
            full_data[key].append(top)

    for key in full_data:
        full_data[key] = np.concatenate(full_data[key])
        print(key, full_data[key].shape, full_data[key][0])
    outpath = ('/mnt/fs3/azhai/center_cropped_embeddings/nearest_neighbors/'
               'ep%i_nearest_list_top%i.npz' % (epoch, k))
    np.savez(open(outpath, 'w'), **full_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract the nearest neighbors of embedding vectors.")
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    mb = load_mb_from_npy('/mnt/fs3/azhai/center_cropped_embeddings/'
                          'classify_nn_clusters_rd2/ep%i_embeddings.npy' % args.epoch)
    get_nearest_list(mb, args.save_dir, args.epoch,
                     start_idx=args.start_idx, end_idx=args.end_idx, topk=100)
