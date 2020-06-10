from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os, sys
import numpy as np
import pdb
from utils import rgb_to_lab, ab_to_Q

sys.path.append('../no_tfutils/')
from vgg_preprocessing import preprocess_image, _aspect_preserving_resize, \
        _central_crop
from resnet_preprocessing import _at_least_x_are_true, \
        _center_crop, _do_scale

EPS = 1e-6


def image_resize(
        crop_image, out_height, out_width, 
        ):
    resize_func = tf.image.resize_area
    image = tf.cast(
            resize_func(
                [crop_image], 
                [out_height, out_width])[0],
            dtype=tf.uint8)
    return image


def RandomSizedCrop_from_jpeg(
        image_str, 
        out_height, 
        out_width, 
        size_minval=0.08,
        ):
    shape = tf.image.extract_jpeg_shape(image_str)
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    crop_max_attempts = 100
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(3. / 4, 4. / 3.),
            area_range=(size_minval, 1.0),
            max_attempts=crop_max_attempts,
            use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, bbox = sample_distorted_bounding_box
    random_image = tf.image.decode_and_crop_jpeg(
            image_str, 
            tf.stack([bbox_begin[0], bbox_begin[1], \
                      bbox_size[0], bbox_size[1]]),
            channels=3)
    bad = _at_least_x_are_true(shape, tf.shape(random_image), 3)
    # central crop if bad
    min_size = tf.minimum(shape[0], shape[1])
    offset_height = tf.random_uniform(
            shape=[],
            minval=0, maxval=shape[0] - min_size + 1,
            dtype=tf.int32
            )
    offset_width = tf.random_uniform(
            shape=[],
            minval=0, maxval=shape[1] - min_size + 1,
            dtype=tf.int32
            )
    bad_image = tf.image.decode_and_crop_jpeg(
            image_str, 
            tf.stack([offset_height, offset_width, \
                      min_size, min_size]),
            channels=3)
    image = tf.cond(
            bad, 
            lambda: bad_image,
            lambda: random_image,
            )
    image = image_resize(
            image, 
            out_height, out_width, 
            )
    image.set_shape([out_height, out_width, 3])
    return image


def RandomSizedCrop(
        image, 
        out_height, 
        out_width, 
        seed_random=0,
        channel_num=3,
        fix_asp_ratio=0,
        size_minval=0.08,
        ):
    shape = tf.shape(image)
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(3. / 4, 4. / 3.),
            area_range=(size_minval, 1.0),
            max_attempts=10,
            use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, bbox = sample_distorted_bounding_box
    random_image = tf.slice(image, bbox_begin, bbox_size)
    bad = _at_least_x_are_true(shape, tf.shape(random_image), 3)

    image = tf.cond(
            bad, 
            lambda: _central_crop(
                [_do_scale(image, max(out_height, out_width))], 
                out_height, out_width)[0],
            lambda: tf.image.resize_bicubic(
                [random_image], 
                [out_height, out_width])[0]
            )
    image = tf.cast(image, tf.uint8)
    image.set_shape([out_height, out_width, 3])
    return image


def RandomBrightness(image, low, high):
    rnd_bright = tf.random_uniform(
            shape=[], 
            minval=low, maxval=high, 
            dtype=tf.float32)
    #rnd_bright = tf.Print(rnd_bright, [rnd_bright], message='Brigh')
    flt_image = tf.cast(image, tf.float32)
    blend_image = flt_image * rnd_bright
    blend_image = tf.maximum(blend_image, 0)
    blend_image = tf.minimum(blend_image, 255)
    image_after = tf.cast(blend_image + EPS, tf.uint8)
    return image_after


def RGBtoGray(flt_image):
    flt_image = tf.cast(flt_image, tf.float32)
    gry_image = flt_image[:,:,0] * 0.299 \
            + flt_image[:,:,1] * 0.587 \
            + flt_image[:,:,2] * 0.114
    gry_image = tf.expand_dims(gry_image, axis=2)
    gry_image = tf.cast(gry_image + EPS, tf.uint8)
    gry_image = tf.cast(gry_image, tf.float32)
    return gry_image


def RandomSaturation(image, low, high):
    rnd_saturt = tf.random_uniform(
            shape=[], 
            minval=low, maxval=high, 
            dtype=tf.float32)
    #rnd_saturt = tf.Print(rnd_saturt, [rnd_saturt], message='Satu')
    flt_image = tf.cast(image, tf.float32)
    gry_image = RGBtoGray(flt_image)
    blend_image = flt_image * rnd_saturt + gry_image * (1-rnd_saturt)
    blend_image = tf.maximum(blend_image, 0)
    blend_image = tf.minimum(blend_image, 255)
    image_after = tf.cast(blend_image + EPS, tf.uint8)
    return image_after


def RandomContrast(image, low, high):
    rnd_contr = tf.random_uniform(
            shape=[], 
            minval=low, maxval=high, 
            dtype=tf.float32)
    #rnd_contr = tf.Print(rnd_contr, [rnd_contr], message='Contr')
    flt_image = tf.cast(image, tf.float32)
    mean_gray = tf.cast(
            tf.cast(
                tf.reduce_mean(RGBtoGray(flt_image)) + 1e-6, 
                tf.uint8), 
            tf.float32)
    blend_image = flt_image * rnd_contr + mean_gray * (1-rnd_contr)
    blend_image = tf.maximum(blend_image, 0)
    blend_image = tf.minimum(blend_image, 255)
    image_after = tf.cast(blend_image + 1e-6, tf.uint8)
    return image_after


def ColorJitter(image, seed_random=0, 
                as_batch=False, shape_undefined=1,
                ):
    order_temp = tf.constant([0,1,2,3], dtype=tf.int32)
    order_rand = tf.random_shuffle(order_temp)
    #order_rand = tf.Print(order_rand, [order_rand], message='Order')

    random_hue_func = tf.image.random_hue

    fn_pred_fn_pairs = lambda x, image: [
            (tf.equal(x, order_temp[0]), \
                    lambda :RandomSaturation(image, 0.6, 1.4)),
            (tf.equal(x, order_temp[1]), \
                    lambda :RandomBrightness(image, 0.6, 1.4)),
            (tf.equal(x, order_temp[2]), \
                    lambda :random_hue_func(image, 0.4)),
            ]
    #default_fn = lambda image: tf.image.random_contrast(image, 0.6, 1.4)
    default_fn = lambda image: RandomContrast(image, 0.6, 1.4)

    def _color_jitter_one(_norm):
        orig_shape = tf.shape(_norm)
        for curr_idx in range(order_temp.get_shape().as_list()[0]):
            _norm = tf.case(
                    fn_pred_fn_pairs(order_rand[curr_idx], _norm), 
                    default=lambda : default_fn(_norm))
        if shape_undefined==0:
            _norm.set_shape(orig_shape)
        return _norm
    if as_batch:
        image = tf.map_fn(_color_jitter_one, image)
    else:
        image = _color_jitter_one(image)
    return image

def ColorJitter_no_rnd(image):
    image = RandomSaturation(image, 0.6, 1.4)
    image = RandomBrightness(image, 0.6, 1.4)
    image = tf.image.random_hue(image, 0.4)
    image = RandomContrast(image, 0.6, 1.4)
    return image


def ColorLighting(image, seed_random=0):
    alphastd = 0.1
    eigval = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
    eigvec = np.array([[-0.5675,  0.7192,  0.4009],
		     [-0.5808, -0.0045, -0.8140],
		     [-0.5836, -0.6948,  0.4203]], dtype=np.float32)

    alpha = tf.random_normal([3, 1], mean=0.0, stddev=alphastd, seed=seed_random)
    rgb = alpha * (eigval.reshape([3, 1]) * eigvec)
    image = image + tf.reduce_sum(rgb, axis=0)

    return image


def ColorNormalize(image):
    transpose_flag = image.get_shape().as_list()[-1] != 3
    if transpose_flag:
        image = tf.transpose(image, [1,2,0])
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - imagenet_mean) / imagenet_std
    if transpose_flag:
        image = tf.transpose(image, [2,0,1])

    return image


def ApplyGray(norm, prob_gray, as_batch=False):
    def _postprocess_gray(im):
        do_gray = tf.random_uniform(
                shape=[], 
                minval=0, 
                maxval=1, 
                dtype=tf.float32)
        def __gray(im):
            gray_im = tf.cast(RGBtoGray(im), tf.uint8)
            gray_im = tf.tile(gray_im, [1,1,3])
            return gray_im
        return tf.cond(
                tf.less(do_gray, prob_gray), 
                lambda: __gray(im), 
                lambda: im)
    if as_batch:
        norm = tf.map_fn(_postprocess_gray, norm, dtype=norm.dtype)
    else:
        norm = _postprocess_gray(norm)
    return norm


def preprocessing_train(
        image, 
        out_height, 
        out_width, 
        seed_random=0
        ):
    #image = tf.Print(image, [tf.shape(image)], message='Init')
    #image = tf.Print(image, [image], message='Convert')

    image = RandomSizedCrop(
            image=image, 
            out_height=out_height,
            out_width=out_width,
            seed_random=seed_random,
            )
    #image = tf.Print(image, [image], message='Rand')
    image = ColorJitter(image, seed_random)

    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    #image = tf.Print(image, [image], message='Jitter')
    image = ColorLighting(image, seed_random)
    #image = tf.Print(image, [image], message='Light')
    image = ColorNormalize(image)
    #image = tf.Print(image, [image], message='Norm')
    image = tf.image.random_flip_left_right(image, seed = seed_random)
    #image = tf.Print(image, [image], message='Flip')

    return image


def preprocessing_val(
        image, 
        out_height, 
        out_width, 
        ):

    image = _aspect_preserving_resize(image, 256)
    image = _central_crop([image], out_height, out_width)[0]
    image.set_shape([out_height, out_width, 3])

    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    image = ColorNormalize(image)

    return image


def preprocessing_th(image, 
        out_height, 
        out_width, 
        seed_random=0,
        is_training=False,
        ):
    if is_training:
        return preprocessing_train(image, out_height, out_width, seed_random)
    else:
        return preprocessing_val(image, out_height, out_width)


def get_resize_scale(height, width, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(
            tf.greater(height, width),
            lambda: smallest_side / width,
            lambda: smallest_side / height)
    return scale


def preprocessing_inst(
        image_string,
        out_height,
        out_width,
        is_train,
        size_minval=0.2,
        skip_color=False,
        skip_gray=False,
        color_no_random=False,
        ):
    def _val_func(image_string):
        shape = tf.image.extract_jpeg_shape(image_string)
        scale = get_resize_scale(shape[0], shape[1], 256)
        cp_height = tf.cast(out_height / scale, tf.int32)
        cp_width = tf.cast(out_width / scale, tf.int32)
        cp_begin_x = tf.cast((shape[0] - cp_height) / 2, tf.int32)
        cp_begin_y = tf.cast((shape[1] - cp_width) / 2, tf.int32)
        bbox = tf.stack([
                cp_begin_x, cp_begin_y, \
                cp_height, cp_width])
        crop_image = tf.image.decode_and_crop_jpeg(
                image_string, 
                bbox,
                channels=3)
        image = image_resize(
                crop_image, 
                out_height, out_width, 
                )

        image.set_shape([out_height, out_width, 3])
        return image

    def _rand_crop(image_string):
        image = RandomSizedCrop_from_jpeg(
                image_string,
                out_height=out_height,
                out_width=out_width,
                size_minval=size_minval,
                )
        return image

    if is_train:
        image = _rand_crop(image_string)
        if not skip_gray:
            image = ApplyGray(image, 0.2)
        if not skip_color:
            image = ColorJitter(image)
        image = tf.image.random_flip_left_right(image)

    else:
        image = _val_func(image_string)

    return image


def col_image_prep(image, is_training):
    crop_size = 224
    size_minval = 0.08
    down_sample = 8

    image = preprocessing_inst(
            image,
            crop_size,
            crop_size,
            is_training,
            size_minval=size_minval,
            skip_color=True,
            skip_gray=True)
    image = tf.cast(image, tf.float32)
    image /= 255
    lab_image = rgb_to_lab(image)
    l_image = lab_image[ :, :, :1]
    l_image = l_image - 50
    ab_image = lab_image[ :, :, 1:]
    ab_image_ss = ab_image[::down_sample, ::down_sample, :]
    Q_label = ab_to_Q(ab_image_ss, soft=not is_training, col_knn=True)
    return l_image, Q_label
