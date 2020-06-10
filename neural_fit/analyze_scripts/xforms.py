'''
List of transformations to apply to the image, heavily inspired by Lucid
source code: https://github.com/tensorflow/lucid/blob/master/lucid/optvis/transform.py
'''

import tensorflow as tf

def jitter(image_tensor, jitter_amount, seed=None):
    '''
    jitter_amount in px
    '''

    shp = tf.shape(image_tensor)
    crop_shape = tf.concat(
        [shp[:-3], shp[-3:-1] - jitter_amount, shp[-1:]],
        0
    )
    crop = tf.random_crop(image_tensor, crop_shape, seed=seed)
    shp = image_tensor.get_shape().as_list()
    mid_shp_changed = [
        shp[-3] - jitter_amount if shp[-3] is not None else None,
        shp[-2] - jitter_amount if shp[-3] is not None else None,
    ]
    crop.set_shape(shp[:-3] + mid_shp_changed + shp[-1:])
    return crop


def pad(image_tensor, pad_amount=12, mode="constant", constant_value=0.5):
    if constant_value == "uniform":
        constant_value_ = tf.random_uniform([], 0, 1)
    else:
        constant_value_ = constant_value

    return tf.pad(
        image_tensor,
        [(0, 0), (pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)],
        mode=mode,
        constant_values=constant_value_,
    )


def random_rotate(image_tensor, angles, units="degrees", seed=None):
    angle = _rand_select(angles, seed=seed)
    angle = _angle2rads(angle, units)
    return tf.contrib.image.rotate(image_tensor, angle)


def random_scale(image_tensor, scales, seed=None):
    scale = _rand_select(scales, seed=seed)
    shp = tf.shape(image_tensor)
    scale_shape = tf.cast(scale * tf.cast(shp[-3:-1], "float32"), "int32")
    return tf.image.resize_bilinear(image_tensor, scale_shape)


def _rand_select(xs, seed=None):
    xs_list = list(xs)
    rand_n = tf.random_uniform((), 0, len(xs_list), "int32", seed=seed)
    return tf.constant(xs_list)[rand_n]


def _angle2rads(angle, units):
    angle = tf.cast(angle, "float32")
    if units.lower() == "degrees":
        angle = 3.14 * angle / 180.
    elif units.lower() in ["radians", "rads", "rad"]:
        angle = angle

    return angle
