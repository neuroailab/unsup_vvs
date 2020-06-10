from __future__ import division, print_function
import os, sys
import tensorflow as tf
import numpy as np
import argparse

# Add parent directory to the system path so we can import from there.
parent, _ = os.path.split(os.getcwd())
if parent not in sys.path:
    sys.path.append(parent)

from tfutils_reader import TfutilsReader, load_model
from embedding_stats import MemoryBank
import image_utils
import model.preprocessing as prep


def calculate_embeddings(sess, model, img_list):
    start = 0
    batch_size = 128
    vecs = []
    while start < len(img_list):
        print('Evaluating embeddings', start, 'to', start + batch_size)
        batch = sess.run(model['embedding'],
            feed_dict={
                model['input_handle']: img_list[start:start+batch_size]
            })
        vecs.append(batch)
        start += batch_size
    vecs = np.concatenate(vecs)
    return vecs


def compute_embeddings():
    parser = argparse.ArgumentParser(
        description="Compute embeddings for a given epoch.")
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--cache_dir', type=str, default='/data/azhai/tmp')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--exp', type=str, default="instance_task/control/full")
    parser.add_argument('--gpu', type=str, required=True,
                        help="GPU(s) to use.")
    args = parser.parse_args()

    dbname, colname, exp_id = args.exp.split('/')
    tr = TfutilsReader(dbname, colname, exp_id,
                       port=27009, cache_dir=args.cache_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    all_vecs = []
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session(graph=g)
        model = load_model(sess, tr, 10009 * args.epoch)
        for i in range(1024):
            print('Processing training file', i)
            img_list = np.load(
                '/mnt/fs3/azhai/center_cropped_imagenet/'
                'train-%05i-of-01024.npy' % i)
            vecs = calculate_embeddings(sess, model, img_list)
            all_vecs.append(vecs)
        sess.close()

    all_vecs = np.concatenate(all_vecs)
    save_path = os.path.join(args.save_dir, 'ep%i_embeddings.npy' % args.epoch)
    np.save(open(save_path, 'w'), all_vecs)

if __name__ == '__main__':
    compute_embeddings()
