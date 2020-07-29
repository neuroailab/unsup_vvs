from __future__ import division

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io
import json
import functools

def load_img_strings_from_tfrecord(path, limit=None):
    record_iter = tf.python_io.tf_record_iterator(path=path)
    ret = []
    for idx, record in enumerate(record_iter):
        if limit is not None and idx >= limit:
            break
        image_example = tf.train.Example()
        image_example.ParseFromString(record)
        img_string = image_example.features.feature['images'].bytes_list.value[0]
        # each image is a jpeg string
        ret.append(img_string)
    return ret


def get_training_img_by_index(idx):
    dataset_dir = '/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label_full_widx'

    f = open(os.path.join(dataset_dir, 'index_info.json'), 'r')
    index_info = json.loads(f.read())
    i = 0
    while i < len(index_info) - 1 and idx >= index_info[i + 1][2]:
        i += 1
    train_name, _, first_idx_in_file = index_info[i]


    jpgs = load_img_strings_from_tfrecord(
        os.path.join(dataset_dir, train_name + '-of-01024'))
    return ImageData(jpgs[idx - first_idx_in_file])


def get_cropping_grid(width, height, crop_size, grid_size):
    '''
    Returns a list of bounding boxes corresponding to a grid of
    different possible croppings.
    '''
    if not isinstance(crop_size, tuple):
        crop_size = (crop_size, crop_size)
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    crop_x, crop_y = crop_size
    nx, ny = grid_size
    xs = np.linspace(0, width - crop_x, num=nx, dtype=np.int32)
    ys = np.linspace(0, height - crop_y, num=ny, dtype=np.int32)
    ret = []
    for x in xs:
        for y in ys:
            ret.append(((x, y), (x + crop_x, y + crop_y)))
    return ret

def _eval_in_session(fn):
    @functools.wraps(fn)
    def wrapper(s, *args, **kwargs):
        # Annoyingly, we will still have a lot of logging info
        # checking for GPUs even though we don't want any in this
        # session. It's fixed in newer versions of tensorflow, see:
        # https://github.com/tensorflow/tensorflow/commit/d61350dfe9568f4efc577f537e93b64980e2696b#diff-3780f0ef44936240abc76c4c42541532
        sess = tf.Session(graph=s._g,
                          config=tf.ConfigProto(device_count={'GPU':0}))
        out_tensor = fn(s, *args, **kwargs)
        ret = sess.run(out_tensor, feed_dict=s._feed)
        sess.close()
        s._feed = {}
        return ret
    return wrapper

class ImageData(object):
    def __init__(self, img_str):
        self.img_str = img_str
        self._g = tf.Graph()
        self._feed = {}

        with self._g.as_default():
            # x, y, height, width
            self._bbox = tf.placeholder(dtype=tf.int32, shape=(4,))

            self._shape = tf.image.extract_jpeg_shape(img_str)
            self._img_tensor = tf.image.decode_jpeg(img_str, channels=3)
            self._cropped = tf.image.decode_and_crop_jpeg(
                img_str, self._bbox, channels=3)
            # TODO: hard-coded 224 by 224 for resnet
            self._cropped_and_scaled = tf.cast(
                tf.image.resize_area([self._cropped], [224, 224])[0],
                dtype=tf.uint8)

        self.shape = self._get_shape()

    @_eval_in_session
    def _get_shape(self):
        with self._g.as_default():
            return tf.image.extract_jpeg_shape(self.img_str)

    @_eval_in_session
    def as_tensor(self):
        return self._img_tensor

    @_eval_in_session
    def as_cropped_tensor(self, topleft, bottomright, scaled=False):
        x1, y1 = topleft
        x2, y2 = bottomright
        self._feed = {self._bbox: [y1, x1, (y2 - y1), (x2 - x1)]}
        if scaled:
            return self._cropped_and_scaled
        else:
            return self._cropped

    def crop_to_center(self):
        '''
        Imitates the cropping used by ResNet for validation.
        '''
        h, w, _ = self.shape
        x, y, dim = 0, 0, min(h, w)
        if h < w:
            x += (w - dim) // 2
        else:
            y += (h - dim) // 2
        return self.as_cropped_tensor((x, y), (x + dim, y + dim), scaled=True)
