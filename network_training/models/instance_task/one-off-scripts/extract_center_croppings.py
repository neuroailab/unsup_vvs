from __future__ import division, print_function
import os, sys
import tensorflow as tf
import numpy as np
import argparse
import time

# Add parent directory to the system path so we can import from there.
parent, _ = os.path.split(os.getcwd())
if parent not in sys.path:
    sys.path.append(parent)

import image_utils
import model.preprocessing as prep
from embedding_stats import augment_img


def extract(file_num, is_train=True):
    imagenet_dir = '/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label_full_widx'
    output_dir = '/mnt/fs3/azhai/center_cropped_imagenet'

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    path_pattern = ('%s/train-%05i-of-01024' if is_train
                    else '%s/validation-%05i-of-00128')

    img_path = path_pattern % (imagenet_dir, file_num)
    print('Reading images from %s' % img_path)
    jpgs = image_utils.load_img_strings_from_tfrecord(img_path)

    img_list = []
    for i, jpg in enumerate(jpgs):
        g = tf.Graph()
        with g.as_default():
            sess = tf.Session(graph=g)
            center_crop = sess.run(prep.resnet_validate(jpg))
            img_list.append(center_crop)
            sess.close()

        if i % 50 == 0:
            print('Cropped', i, 'images')

    img_list = np.array(img_list)
    outpath = path_pattern % (output_dir, file_num)
    print('Saving center cropped output to %s' % outpath)
    np.save(open(outpath, 'w'), img_list)


def extract_augmentations(file_num, num_augments, batch_num):
    imagenet_dir = '/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label_full_widx'
    output_dir = '/mnt/fs3/azhai/center_cropped_imagenet/augmentations'

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    jpgs = image_utils.load_img_strings_from_tfrecord(
        '%s/train-%05i-of-01024' % (imagenet_dir, file_num)
    )

    img_list = []
    for i, jpg in enumerate(jpgs):
        augs, _ = augment_img(jpg, num_augments)
        img_list.append(np.array(augs))
        if i % 10 == 0:
            print('Augmented', i, 'images')
    img_list = np.array(img_list)
    print(img_list.shape) # (num imgs, num augments, 224, 224, 3)

    outpath = '%s/train%i_augments%i.npy' % (output_dir, file_num, batch_num)
    print('Saving augmentations to', outpath)
    np.save(open(outpath, 'w'), img_list)


if __name__ == '__main__':
    # extract_augmentations(file_num=int(sys.argv[1]),
    #                       num_augments=10, batch_num=int(sys.argv[2]))

    parser = argparse.ArgumentParser(
        description="Extracted center-cropped images as numpy arrays.")
    parser.add_argument('--filenum', type=int)
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    extract(args.filenum, is_train=(not args.validation))
