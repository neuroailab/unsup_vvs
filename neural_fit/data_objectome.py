from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import cPickle
import pdb
import copy
import pandas as pd
from PIL import Image

OBJECTOME_DATA_LEN = 2400
V1_CADENA_DATA_LEN = 6249


class ObjectomeZip(object):
    """
    Load from unzipped folder
    """

    def __init__(self, data_path, file_suffix='.png', **kwargs):
        csv_path = os.path.join(data_path, 'stimulus_set.csv')
        stimuli = pd.read_csv(csv_path)
        image_paths = [
                os.path.join(data_path, image_id + file_suffix) \
                for image_id in stimuli['image_id']
                ]
        image_paths = filter(lambda x: os.path.exists(x), image_paths)
        self.image_paths = image_paths

    def __call__(self):
        for image_path in self.image_paths:
            im_frame = Image.open(image_path)
            np_frame = np.array(im_frame)
            yield np_frame


def dataset_func(
        batch_size,
        dataset_type='objectome',
        cadena_im_size=40,
        **generator_kwargs):
    if dataset_type == 'objectome':
        obj_gen = ObjectomeZip(**generator_kwargs)
    elif dataset_type == 'v1_cadena':
        obj_gen = ObjectomeZip(file_suffix='.jpg', **generator_kwargs)
    else:
        raise NotImplementedError()

    ds = tf.data.Dataset.from_generator(obj_gen, (tf.uint8))

    # Change content in dataset to a dict format
    def _to_dict(value):
        dict_value = {
                'images': value,
                }
        return dict_value
    ds = ds.map(
            _to_dict,
            num_parallel_calls=48)

    # Resize the image to 224*224, and color normalize it
    def _normalize_image(value):
        image = value['images']
        image.set_shape([224, 224, 3])
        image = tf.cast(image, tf.float32)
        image /= 255
        value['images'] = image
        return value

    def _cadena_subsample_normalize_image(value):
        image = value['images']
        image.set_shape([140, 140])
        image = tf.cast(image, tf.float32)
        image /= 255
        image = image[30:110, 30:110]
        image = tf.tile(tf.expand_dims(image, axis=2), [1, 1, 3])
        image = tf.image.resize_images(image, [cadena_im_size, cadena_im_size])
        value['images'] = image
        return value

    if dataset_type == 'objectome':
        ds = ds.map(
                _normalize_image,
                num_parallel_calls=48)
    elif dataset_type == 'v1_cadena':
        ds = ds.map(
                _cadena_subsample_normalize_image,
                num_parallel_calls=48)

    # Make the iterator
    ds = ds.repeat()
    ds = ds.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))
    value = ds.make_one_shot_iterator().get_next()
    return value


if __name__=='__main__':
    #obj_zip = ObjectomeZip('/home/chengxuz/objectome')
    #data_iter = dataset_func(60, data_path='/home/chengxuz/objectome')
    data_iter = dataset_func(
            60, 
            dataset_type='v1_cadena',
            data_path='/mnt/fs4/chengxuz/v1_cadena_related/images')

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.Session()
    test_image = sess.run(data_iter)
    test_image = test_image['images']
    print(test_image.shape)
