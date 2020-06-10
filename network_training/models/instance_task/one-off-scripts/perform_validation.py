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
from embedding_stats import MemoryBank, calculate_embeddings
from utils import DATA_LEN_IMAGENET_FULL


class Validator(object):
    def __init__(self, sess, model, memory_bank=None, k=200):
        self.k = k
        self.model = model
        self.labels = np.load(
            '/mnt/fs3/azhai/center_cropped_imagenet/all_labels.npy')

        if memory_bank is None:
            # Assume model has loaded memory bank
            self.memory_bank = tf.transpose(model['memory_bank'])
        else:
            self.memory_bank = tf.get_variable(
                'embeddings', shape=(128, len(self.labels)),
                dtype=tf.float32,
                initializer=tf.zeros_initializer, trainable=False)
            sess.run(tf.assign(self.memory_bank, memory_bank.T))

        all_dps = tf.matmul(self.model['embedding'], self.memory_bank)
        self.top = tf.nn.top_k(all_dps, k=self.k, sorted=False)

    def get_knn_probs(self, sess, img_list, labels):
        top_values, top_indices = sess.run(self.top, feed_dict={
            self.model['input_handle']: img_list
        })
        top_labels = np.take(self.labels, top_indices)
        assert top_labels.shape == top_indices.shape

        exponents = np.exp(top_values / 0.07)
        # row_sums = np.sum(exponents, axis=1)
        # top_probs = exponents / row_sums[:,None]

        ret = []
        for i in range(len(img_list)):
            weights = np.zeros(1000)
            for weight, label in zip(exponents[i], top_labels[i]):
                weights[label] += weight
            ret.append(1. if np.argmax(weights) == labels[i] else 0.)
        return np.array(ret)


class ValidationLoader(object):
    def __init__(self):
        self.cur_idx = 0
        self.val_labels = np.load('/mnt/fs3/azhai/center_cropped_imagenet/validation_labels.npy')
        self.imgs_left = []
        self.next_file_id = 0

    def take(self, num):
        while len(self.imgs_left) < num:
            next_imgs = np.load(
                '/mnt/fs3/azhai/center_cropped_imagenet/'
                'validation-%05i-of-00128.npy' % self.next_file_id)
            self.next_file_id += 1
            if len(self.imgs_left) == 0:
                self.imgs_left = next_imgs
            else:
                self.imgs_left = np.concatenate([self.imgs_left, next_imgs])

        imgs = self.imgs_left[:num]
        labels = self.val_labels[self.cur_idx:self.cur_idx + num]
        self.cur_idx += num
        self.imgs_left = self.imgs_left[num:]
        return imgs, labels


EXP_MAPPINGS = {
    "instance_task/control/full": "/mnt/fs3/azhai/center_cropped_embeddings/ep%i_embeddings.npy",
    "instance_task/clustering/static_nn2_from_ep220": "/mnt/fs3/azhai/center_cropped_embeddings/static_nn2/ep%i_embeddings.npy",
}

# e.g. python perform_validation --gpu 0 --epoch 200 --exp instance_task/control/full --k 1
def validate():
    parser = argparse.ArgumentParser(
        description="Evaluate on validation set.")
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--gpu', type=str, help="GPU(s) to use.", required=True)
    parser.add_argument('--exp', type=str, default="instance_task/control/full")
    parser.add_argument('--k', type=int, default=200)
    parser.add_argument('--cache_dir', type=str, default="/data/azhai/tmp")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.exp in EXP_MAPPINGS:
        embeddings_path = EXP_MAPPINGS[args.exp]
    else:
        embeddings_path = os.path.join(
            '/mnt/fs3/azhai/center_cropped_embeddings',
            args.exp.split('/')[-1], 'ep%i_embeddings.npy')
    mb = np.load(embeddings_path % args.epoch)

    dbname, collname, exp_id = args.exp.split('/')
    tr = TfutilsReader(dbname, collname, exp_id,
                       port=27009, cache_dir=args.cache_dir)
    sess = tf.Session()
    model = load_model(sess, tr, 10009 * args.epoch,
                       load_mem_bank=True,
                       mem_bank_shape=(DATA_LEN_IMAGENET_FULL, 128))

    validator = Validator(sess, model, memory_bank=mb, k=args.k)
    validation_loader = ValidationLoader()

    correct_sum = 0.0
    for i in range(500):
        print('Evaluating', i * 100)
        imgs, labels = validation_loader.take(100)
        probs = validator.get_knn_probs(sess, imgs, labels)
        correct_sum += np.sum(probs)
        print('perf:', correct_sum / validation_loader.cur_idx)


if __name__ == '__main__':
    validate()
