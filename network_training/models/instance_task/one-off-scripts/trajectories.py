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


class MemoryTrajectories(object):
    def __init__(self, tfu_reader, save_dir):
        self.tr = tfu_reader
        self.save_dir = save_dir
        self.idx_sets = {}

    def add_idx_set(self, name, idxs):
        self.idx_sets[name] = idxs

    def extract_memory_trajectories(self, epochs):
        trajectories = {}
        for name in self.idx_sets:
            trajectories[name] = []

        for ep in epochs:
            print('Extracting epoch', ep)
            mb = MemoryBank.load(self.tr, 10009 * ep, 1)
            for name, idxs in self.idx_sets.items():
                trajectories[name].append(mb.memory_vecs[idxs])

        for name, vecs in trajectories.items():
            path = os.path.join(self.save_dir, '%s.npy' % name)
            vecs = np.stack(vecs) # (num epochs, num idxs, embedding dim)
            np.save(open(path, 'w'), vecs)
            print('Saved %s to %s' % (name, path))


def extract_memories():
    parser = argparse.ArgumentParser(
        description="Calculate how memory values change over many epochs.")
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    tr = TfutilsReader('instance_task', 'control', 'full',
                       port=27009, cache_dir=args.cache_dir)
    mb = MemoryBank.load(tr, 10009 * 1, 1)

    traj = MemoryTrajectories(tr, args.save_dir)
    traj.add_idx_set('class_76_tarantula', mb.class_to_idxs[76])
    traj.add_idx_set('class_207_golden_retriever', mb.class_to_idxs[207])
    traj.add_idx_set('class_486_cello', mb.class_to_idxs[486])
    traj.add_idx_set('class_530_digital_clock', mb.class_to_idxs[530])
    traj.add_idx_set('class_614_kimono', mb.class_to_idxs[614])
    traj.add_idx_set('class_698_palace', mb.class_to_idxs[698])
    traj.add_idx_set('class_752_racquet', mb.class_to_idxs[752])
    traj.add_idx_set('class_805_soccer_ball', mb.class_to_idxs[805])
    traj.add_idx_set('class_850_teddy_bear', mb.class_to_idxs[850])
    traj.add_idx_set('class_937_broccoli', mb.class_to_idxs[937])
    traj.add_idx_set('class_987_corn', mb.class_to_idxs[987])
    traj.add_idx_set('first10000', range(10000))

    traj.extract_memory_trajectories(range(1, 201))


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


def extract_center_crop_trajectories():
    parser = argparse.ArgumentParser(
        description="Calculate how embedding values change over many epochs.")
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--valid_num', type=int, default=None,
                        help="Which validation file to process")
    parser.add_argument('--train_num', type=int, default=None,
                        help="Which training file to process")
    parser.add_argument('--gpu', type=str,
                        help="GPU(s) to use.")
    args = parser.parse_args()

    if args.train_num is None and args.valid_num is None:
        raise Exception('Must specify either a train or validation file number!')
    if args.train_num is not None:
        imgs_path = ('/mnt/fs3/azhai/center_cropped_imagenet/'
                     'train-%05i-of-01024.npy' % args.train_num)
    else:
        imgs_path = ('/mnt/fs3/azhai/center_cropped_imagenet/'
                     'validation-%05i-of-00128.npy' % args.valid_num)
    if args.train_num is not None:
        outpath = ('/mnt/fs3/azhai/trajectories/'
                   'traj_train%i.npy' % args.train_num)
    else:
        outpath = ('/mnt/fs3/azhai/trajectories/'
                   'traj_valid%i.npy' % args.valid_num)

    tr = TfutilsReader('instance_task', 'control', 'full',
                       port=27009, cache_dir=args.cache_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # get imgs, compute the embedding
    img_list = np.load(imgs_path)
    all_vecs = []
    for epoch in range(1, 201):
        print('Processing epoch', epoch)
        g = tf.Graph()
        with g.as_default():
            sess = tf.Session(graph=g)
            model = load_model(sess, tr, 10009 * epoch)
            vecs = calculate_embeddings(sess, model, img_list)
            all_vecs.append(vecs)
            sess.close()
    all_vecs = np.stack(all_vecs)
    print(all_vecs.shape)
    np.save(open(outpath, 'w'), all_vecs)
    return all_vecs


def extract_augmentation_trajectories():
    aug_embeddings_dir = '/mnt/fs3/azhai/augmentation_cloud/canonical_train0_augments'
    file_pattern = 'ep%i_train0_aug_embeddings.npy'

    trajs = []
    for epoch in range(1, 201):
        print('Processing epoch', epoch)
        input_path = os.path.join(aug_embeddings_dir, (file_pattern % epoch))
        epoch_embeddings = np.load(input_path)
        trajs.append(epoch_embeddings[:10])
    trajs = np.stack(trajs, axis=1)
    print(trajs.shape)
    outpath = '/mnt/fs3/azhai/trajectories/traj_first10_augments.npy'
    np.save(open(outpath, 'w'), trajs)


if __name__ == '__main__':
    extract_augmentation_trajectories()
    #extract_center_crop_trajectories()
    #extract_memories()
