from __future__ import division, print_function
import os, sys
import numpy as np
import tensorflow as tf
import argparse

# Add parent directory to the system path so we can import from there.
parent, _ = os.path.split(os.getcwd())
if parent not in sys.path:
    sys.path.append(parent)

import image_utils
from image_utils import ImageData
import tfutils_reader
from tfutils_reader import TfutilsReader
import model.preprocessing as prep
from embedding_stats import augment_img, calculate_embeddings


def calculate_embeddings_from_precomputed_augments():
    parser = argparse.ArgumentParser(
        description="Generate embeddings from precomputed data augmentations.")
    parser.add_argument('--gpu', type=str,
                        help="GPU(s) to use.")
    parser.add_argument('--output_dir', type=str,
                        help="Directory to save the resulting numpy array(s).")
    parser.add_argument('--start_epoch', type=int,
                        help=("Starting epoch to process."))
    parser.add_argument('--num_epochs', type=int, default=1,
                        help=("How many epochs to process."))
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sess = tf.Session()
    tr = TfutilsReader('instance_task', 'control', 'full',
                       port=27009, cache_dir='/data/azhai/tmp')

    models = []
    for i in range(args.num_epochs):
        with tf.variable_scope("model_%i" % (args.start_epoch + i)):
            model = tfutils_reader.load_model(sess, tr, 10009 * (args.start_epoch + i))
            models.append(model)

    embeddings = [[] for _ in range(args.num_epochs)]
    for i in range(200):
        aug_imgs = np.load('/mnt/fs3/azhai/center_cropped_imagenet/augmentations/'
                           'train0_augments%i.npy' % i)
        n_examples = aug_imgs.shape[0]
        aug_imgs = np.concatenate(aug_imgs)

        for j in range(args.num_epochs):
            aug_vecs = calculate_embeddings(sess, models[j], aug_imgs,
                                            print_progress=False)
            aug_vecs = np.split(aug_vecs, n_examples, axis=0)
            embeddings[j].append(aug_vecs)
            print('Processed augmentation set %i for epoch %i'
                  % (i, args.start_epoch + j))

    for j in range(args.num_epochs):
        embeddings[j] = np.concatenate(embeddings[j], axis=1)
        print(embeddings[j].shape)
        epoch = args.start_epoch + j
        output_path = os.path.join(args.output_dir,
                                   ('ep%i_train0_aug_embeddings.npy' % epoch))
        print('Saving to', output_path)
        np.save(open(output_path, 'w'), embeddings[j])


def calculate_aug_embeddings_from_tfrecord():
    parser = argparse.ArgumentParser(
        description="Generate embedding outputs after data augmentation.")
    parser.add_argument('--gpu', type=str,
                        help="GPU(s) to use.")
    parser.add_argument('--input_path', type=str,
                        help="Path to an input TFRecord file with the images")
    parser.add_argument('--output_path', type=str,
                        help="Path to save the resulting numpy array.")
    parser.add_argument('--epoch_num', type=int,
                        help="Which epoch to load the model weights from.")
    parser.add_argument('--num_samples', type=int, default=2000,
                        help="Samples to take (if augmentation is random).")
    args = parser.parse_args()

    tr = TfutilsReader('instance_task', 'control', 'full',
                       port=27009, cache_dir='/data/azhai/tmp')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sess = tf.Session()
    model = tfutils_reader.load_model(sess, tr, 10009 * args.epoch_num)

    # # Sample input paths
    # '/mnt/fs1/azhai/image_label_widx_first100/validation-00002-of-00128',
    # '/mnt/fs1/azhai/image_label_widx_first100/train-00001-of-01024',

    jpgs = image_utils.load_img_strings_from_tfrecord(args.input_path)
    cc_list = []
    all_aug_embeddings = []
    for i, img_str in enumerate(jpgs):
        # print('Processing image', i)
        augs, center_crop = augment_img(img_str, args.num_samples)
        cc_list.append(center_crop)
        aug_embeddings = calculate_embeddings(sess, model, augs)
        assert aug_embeddings.shape == (args.num_samples, 128)
        all_aug_embeddings.append(aug_embeddings)

    cc_embeddings = calculate_embeddings(sess, model, cc_list)
    assert cc_embeddings.shape == (len(jpgs), 128)
    all_aug_embeddings = np.stack(all_aug_embeddings, axis=0)
    assert all_aug_embeddings.shape == (len(jpgs), args.num_samples, 128)
    np.savez(open(args.output_path, 'w'),
             center_crop_embeddings=cc_embeddings,
             augmented_embeddings=all_aug_embeddings)


if __name__ == '__main__':
    calculate_embeddings_from_precomputed_augments()
    #calculate_aug_embeddings_from_tfrecord()
