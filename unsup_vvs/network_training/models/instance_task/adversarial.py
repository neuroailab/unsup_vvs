from __future__ import division, print_function
import os, sys
import numpy as np
import tensorflow as tf
import argparse
import scipy.misc

import image_utils
from image_utils import ImageData
import tfutils_reader
from tfutils_reader import TfutilsReader
import model.preprocessing as prep


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


def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    #print [str(i.name) for i in not_initialized_vars] # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def setup_adversarial_training(
        sess, target_image, starting_image,
        epoch_num, use_augmentations=True, batch_size=16):
    tr = TfutilsReader('instance_task', 'control', 'full',
                       port=27009, cache_dir='/data/azhai/tmp')
    delta_var = tf.get_variable(
        "input_image", initializer=tf.zeros(starting_image.shape),
        dtype=tf.float32)
    input_img = tf.cast(tf.convert_to_tensor(starting_image), tf.float32) + delta_var

    if use_augmentations:
        input_handle = crop_and_resize_img(input_img, batch_size)
    else:
        batch_size = 1
        input_handle = tf.expand_dims(input_img, axis=0)

    model = tfutils_reader.load_model(
        sess, tr, 10009 * epoch_num, input_handle=input_handle
    )
    model_output = model['embedding']
    model_input = model['input_handle']

    target_vec = sess.run(model_output,
        feed_dict={
            model_input: np.stack([target_image for _ in range(batch_size)])
        })
    target_vec = tf.cast(tf.convert_to_tensor(target_vec), dtype=tf.float32)
    loss = tf.nn.l2_loss(model_output - target_vec) / batch_size
    # # Optimize towards minimizing self-loss
    # output_mean = tf.reduce_mean(model_output, axis=0)
    # output_mean_batch = tf.stack([output_mean for _ in range(batch_size)])
    # loss = tf.nn.l2_loss(model_output - output_mean_batch)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=500, # TODO: look into different optimization params
        momentum=0.96)
    grads = optimizer.compute_gradients(
        loss, var_list=[delta_var])
    mini_act = optimizer.apply_gradients(grads)

    initialize_uninitialized(sess)
    return delta_var, loss, mini_act, model


def add_noise(arr, ratio):
    whitenoise = 256 * np.random.rand(*arr.shape)
    return (1 - ratio) * arr + ratio * whitenoise


def main():
    parser = argparse.ArgumentParser(
        description="Generate embedding outputs after data augmentation.")
    parser.add_argument('--gpu', type=str, required=True,
                        help="GPU(s) to use.")
    parser.add_argument('--epoch_num', type=int, required=True,
                        help="Which epoch to load the model weights from.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to store the resulting image.")
    parser.add_argument('--img_idxs', type=str, required=True,
                        help="a,b where we perturb image a to disguise it as image b")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    a, b = args.img_idxs.split(',')
    a, b = int(a), int(b)

    imgs = np.load('/mnt/fs3/azhai/center_cropped_imagenet/train-00000-of-01024.npy')
    goal_img = imgs[b]
    another_img = imgs[a]

    # goal_img = image_utils.get_training_img_by_index(b).crop_to_center()
    # print('goal img', goal_img.shape)
    # #another_img = add_noise(goal_img, 0.95)
    # another_img = image_utils.get_training_img_by_index(a).crop_to_center()
    # #another_img = image_utils.get_training_img_by_index(b).as_tensor()
    # #another_img = np.full(goal_img.shape, 128.0)

    sess = tf.Session()
    batch_size = 32
    img_delta, loss, mini_act, model = setup_adversarial_training(
        sess, goal_img, another_img, args.epoch_num,
        use_augmentations=False,
        batch_size=batch_size
    )

    loss_val = 1.0
    i = 0
    eps = 8.0
    with tf.control_dependencies([mini_act]):
        lower = tf.cast(tf.minimum(another_img + 0.0, eps), dtype=tf.float32)
        upper = tf.cast(tf.minimum(255.0 - another_img, eps), dtype=tf.float32)
        new_delta = tf.maximum(img_delta, -lower)
        new_delta = tf.minimum(new_delta, upper)
        update_delta = tf.assign(img_delta, new_delta)
    while loss_val > 0.00001 and i < 10000:
        _, loss_val, ud = sess.run([mini_act, loss, update_delta])
        single_loss_sq = 2 * loss_val
        implied_dp = 1 - single_loss_sq/2
        #print('ud bounds', ud.shape, np.max(ud), np.min(ud))
        print('%d: loss %f, implied dot product %f' % (i, loss_val, implied_dp))
        i += 1

    final_img = sess.run(another_img + img_delta)
    print('final img', final_img.shape)

    res1 = sess.run(model['embedding'], feed_dict={model['input_handle']: [goal_img]})
    res2 = sess.run(model['embedding'], feed_dict={model['input_handle']: [final_img]})

    print('delta size', np.max(final_img - another_img), np.max(another_img - final_img))
    print('error', np.linalg.norm(res1 - res2))

    np.save(args.output_path, final_img)
    #scipy.misc.imsave(args.output_path, np.abs(final_img - another_img))

if __name__ == '__main__':
    main()
