from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os, sys
import numpy as np
import scipy.misc
import time
import argparse

import image_utils
import tfutils_reader
from tfutils_reader import TfutilsReader
from instance_stats import analysis


def sample_bbox(orig_shape):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            orig_shape,
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(3. / 4, 4. / 3.),
            area_range=(0.2, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, bbox = sample_distorted_bounding_box
    return bbox[0][0]


def crop_and_resize_img(img, batch_size):
    bboxes = [sample_bbox(img.shape) for i in range(batch_size)]
    bboxes = tf.stack(bboxes, axis=0)
    img_batch = tf.stack([img for _ in range(batch_size)])
    print(img_batch.get_shape().as_list())
    print(bboxes.get_shape().as_list())
    return tf.image.crop_and_resize(
        img_batch,
        boxes=bboxes,
        box_ind=tf.constant(range(batch_size)),
        crop_size=tf.constant([224,224]))


def compute_gradients(img_idx, coord):
    outpath = ('/mnt/fs3/azhai/derivatives/epoch100/'
               'idx%i/d%i.npy' % (args.img_idx, coord))
    img = image_utils.get_training_img_by_index(img_idx)
    img = img.as_tensor()
    print(img.shape)

    sess = tf.Session()
    input_var = tf.get_variable(
        "input_image", initializer=tf.zeros(img.shape),
        dtype=tf.float32)
    sess.run(tf.assign(input_var, img))

    batch_size = 64
    cropped = crop_and_resize_img(input_var, batch_size)
    print(cropped.shape, cropped.dtype)

    tr = TfutilsReader('instance_task', 'control', 'full',
                       port=27009, cache_dir='/data/azhai/tmp')
    epoch_num = 100
    model = tfutils_reader.load_model(
        sess, tr, 10009 * epoch_num,
        input_handle=cropped
    )
    model_output = model['embedding']
    model_input = model['input_handle']

    N = 16
    total_grad = np.zeros_like(img, dtype=np.float32)

    last_time = time.time()
    for i in range(N):
        tot = tf.reduce_sum(model_output, axis=0)
        grad = sess.run(tf.gradients(tot[coord], input_var))
        grad = grad[0]
        assert grad.shape == total_grad.shape
        total_grad += grad

        print('Processed trial', i, 'elapsed =', time.time() - last_time)
        last_time = time.time()

    total_grad /= (N * batch_size)
    np.save(open(outpath, 'w'), total_grad / N)


def save_tensor_to_img(tensor, outpath):
    scipy.misc.imsave(outpath, np.absolute(tensor))


def do_pca(img_idx, num_components=1):
    shape = None
    derivs = []
    for i in range(128):
        row = np.load('/mnt/fs3/azhai/derivatives/epoch100/idx%i/d%i.npy'
                      % (img_idx, i))
        shape = row.shape
        row = np.ndarray.flatten(row)
        derivs.append(row)
        print('loaded row', i)
    derivs = np.stack(derivs)

    weights, _, topvecs = analysis.pca_analysis(derivs)
    print(topvecs[0].shape)

    for i in range(num_components):
        vec = topvecs[i].reshape(shape)
        outpath = ('/mnt/fs3/azhai/derivatives/epoch100/idx%i/grad%i.jpg'
                   % (img_idx, i))
        save_tensor_to_img(vec, outpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate averaged gradients over random croppings.")
    parser.add_argument('--gradient_pca', action='store_true')
    parser.add_argument('--num_components', type=int)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--coord', type=int)
    parser.add_argument('--img_idx', type=int)
    args = parser.parse_args()

    if args.gradient_pca:
        do_pca(args.img_idx, num_components=args.num_components)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        compute_gradients(args.img_idx, args.coord)
