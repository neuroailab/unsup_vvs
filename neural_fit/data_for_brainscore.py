import os, sys
import numpy as np
import tensorflow as tf
from PIL import Image
import pdb


class ImagesFromFolder(object):
    """
    Load from a folder containing images
    """
    def __init__(self, folder_dir, image_list, **kwargs):
        image_paths = [
                os.path.join(folder_dir, image_name) \
                for image_name in image_list]
        image_paths = filter(lambda x: os.path.exists(x), image_paths)
        self.image_paths = image_paths

    def __call__(self):
        for image_path in self.image_paths:
            im_frame = Image.open(image_path)
            np_frame = np.array(im_frame)
            yield np_frame


def dataset_func(
        batch_size,
        img_out_size=224,
        data_norm_type='standard',
        **generator_kwargs):
    obj_gen = ImagesFromFolder(**generator_kwargs)
    ds = tf.data.Dataset.from_generator(obj_gen, (tf.uint8)).repeat()

    # Change content in dataset to a dict format
    def _to_dict(value):
        dict_value = {'images': value}
        return dict_value
    ds = ds.map(
            _to_dict,
            num_parallel_calls=48)

    # Resize the image to 224*224, and color normalize it
    def _normalize_image(value):
        image = value['images']
        image.set_shape([256, 256, 3])
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_images(
                image, [img_out_size, img_out_size])
        value['images'] = image
        return value

    def _cadena_subsample_normalize_image(value):
        image = value['images']
        image.set_shape([140, 140])
        image = tf.cast(image, tf.float32)
        image = image[30:110, 30:110]
        image = tf.tile(tf.expand_dims(image, axis=2), [1, 1, 3])
        image = tf.image.resize_images(image, [img_out_size, img_out_size])
        value['images'] = image
        return value

    if data_norm_type == 'standard':
        ds = ds.map(
                _normalize_image,
                num_parallel_calls=48)
    elif data_norm_type == 'v1_cadena':
        ds = ds.map(
                _cadena_subsample_normalize_image,
                num_parallel_calls=48)
    else:
        raise NotImplementedError('Data normalization type not supported!')

    # Make the iterator
    ds = ds.batch(batch_size)
    value = ds.make_one_shot_iterator().get_next()
    return value
