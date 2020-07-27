# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import resnet_preprocessing


class PBRSceneNetDepthInput(object):
    """Generates PBRNet and Scenenet data input_pbrscenent for training or evaluation.

    The training data is assumed to be in TFRecord format with keys as specified
    in the dataset_parser below, sharded across 366 files, named sequentially:
        train-0
        train-1
        ...
        train-365

    The validation data is in the same format but sharded in 50 files.

    Args:
      is_training: `bool` for whether the input is for training
      data_dir: `str` for the directory of the training and validation data
      num_cores: `int` for the number of TPU cores
    """

    def __init__(self, is_training, pbr_dir, scene_dir, num_cores=8):
        # self.image_preprocessing_fn = resnet_preprocessing.full_imagenet_augment
        self.image_preprocessing_fn = resnet_preprocessing.depth_image_preprocess_image
        self.depth_preprocessing_fn = resnet_preprocessing.depth_preprocess_image
        self.is_training = is_training
        self.pbr_dir = pbr_dir
        self.scene_dir = scene_dir
        self.num_cores = num_cores
        self.training_data_num = 366 + 843

    def dataset_parser(self, value):
        """Parse an ImageNet record from a serialized string Tensor."""
        keys_to_features = {
            'mlt': tf.FixedLenFeature((), tf.string, ''),
            'depth': tf.FixedLenFeature((), tf.string, '')
        }

        parsed = tf.parse_single_example(value, keys_to_features)
        print("parsed example", parsed)

        original_image = tf.reshape(parsed['mlt'], shape=[])
        original_image = tf.image.decode_png(original_image, channels=3)
        original_image = tf.image.convert_image_dtype(original_image, dtype=tf.float32)
        original_image.set_shape((240, 320, 3))

        depth_image = tf.reshape(parsed['depth'], shape=[])
        depth_image = tf.image.decode_png(depth_image, dtype=tf.uint16)
        depth_image.set_shape((240, 320, 1))
        depth_image = tf.cast(depth_image, tf.int32)

        original_image, depth_image = self.image_preprocessing_fn(
            original_image=original_image,
            depth_image=depth_image,
            is_training=self.is_training,
            image_height=240,
            image_width=320,
        )
        depth_image = self.depth_preprocessing_fn(
            image=depth_image,
        )
        return original_image, depth_image

    def input_fn(self, params):
        """Input function which provides a single batch for train or eval.

        Args:
          params: `dict` of parameters passed from the `TPUEstimator`.
              `params['batch_size']` is always provided and should be used as the
              effective batch size.

        Returns:
          A (images, labels) tuple of `Tensor`s for a batch of samples.
        """
        # Retrieves the batch size for the current shard. The # of shards is
        # computed according to the input pipeline deployment. See
        # tf.contrib.tpu.RunConfig for details.

        batch_size = params['batch_size']

        # Shuffle the filenames to ensure better randomization.
        pbr_file_pattern = os.path.join(
            self.pbr_dir, 'train-*' if self.is_training else 'validation-*')
        dataset = tf.data.Dataset.list_files(pbr_file_pattern)

        print("dataset", dataset)

        scene_file_pattern = os.path.join(
            self.scene_dir, 'train-*' if self.is_training else 'validation-*')
        dataset_tmp = tf.data.Dataset.list_files(scene_file_pattern)

        print("dataset_tmp", dataset_tmp)

        dataset.concatenate(dataset_tmp)

        print("dataset", dataset)


        if self.is_training:
            dataset = dataset.shuffle(buffer_size=self.training_data_num)  # 1024 files in dataset
            dataset = dataset.repeat()

        def fetch_dataset(filename):
            buffer_size = 1024 * 1024 * 1024  # 1024 MiB per file
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset

        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                fetch_dataset, cycle_length=self.num_cores, sloppy=True))

        dataset = dataset.shuffle(batch_size * 20)

        print("before parsing", dataset)
        dataset = dataset.map(
            self.dataset_parser,
            num_parallel_calls=64)
        print("after parsing", dataset)
        dataset = dataset.prefetch(batch_size)

        # For training, batch as usual. When evaluating, prevent accidentally
        # evaluating the same image twice by dropping the final batch if it is less
        # than a full batch size. As long as this validation is done with
        # consistent batch size, exactly the same images will be used.
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))

        dataset = dataset.prefetch(2)  # Prefetch overlaps in-feed with training

        original_images, depth_images = dataset.make_one_shot_iterator().get_next()
        print("original image and depth type:")
        print(original_images.shape)
        print(depth_images.shape)

        return original_images, depth_images

    def postproc_flag(self, images,
                      NOW_SIZE1=256,
                      NOW_SIZE2=256,
                      seed_random=0,
                      curr_batch_size=None,
                      with_noise=0,
                      noise_level=10,
                      with_flip=0,
                      is_normal=0,
                      eliprep=0,
                      thprep=0,
                      sub_mean=0,
                      mean_path=None,
                      color_norm=0,
                      size_vary_prep=0,
                      with_color_noise=0,
                      shape_undefined=0,
                      size_minval=0.08,
                      sm_full_size=0,  # sm_add
                      ):

        if curr_batch_size == None:
            curr_batch_size = self.batch_size

        orig_dtype = images.dtype
        norm = tf.cast(images, tf.float32)

        if eliprep == 1:
            def prep_each(norm_):
                _RESIZE_SIDE_MIN = 256
                _RESIZE_SIDE_MAX = 512

                if self.group == 'train':
                    im = preprocess_image(norm_, self.crop_size, self.crop_size, is_training=True,
                                          resize_side_min=_RESIZE_SIDE_MIN,
                                          resize_side_max=_RESIZE_SIDE_MAX)
                else:
                    im = preprocess_image(norm_, self.crop_size, self.crop_size, is_training=False,
                                          resize_side_min=_RESIZE_SIDE_MIN,
                                          resize_side_max=_RESIZE_SIDE_MAX)

                return im

            crop_images = tf.map_fn(prep_each, norm)
        elif thprep == 1:
            def prep_each(norm_):
                im = preprocessing_th(norm_, self.crop_size, self.crop_size,
                                      is_training=self.group == 'train', seed_random=seed_random)
                return im

            crop_images = prep_each(images)
            crop_images = tf.expand_dims(crop_images, axis=0)
        elif sm_full_size == 1:
            if with_color_noise == 1 and self.group == 'train':
                order_temp = tf.constant([0, 1, 2], dtype=tf.int32)
                order_rand = tf.random_shuffle(order_temp, seed=seed_random)

                fn_pred_fn_pairs = lambda x, image: [
                    (tf.equal(x, order_temp[0]),
                     lambda: tf.image.random_saturation(image, 0.6, 1.4, seed=seed_random)),
                    (tf.equal(x, order_temp[1]), lambda: tf.image.random_brightness(image, 0.4, seed=seed_random)),
                ]
                default_fn = lambda image: tf.image.random_contrast(image, 0.6, 1.4, seed=seed_random)

                def _color_jitter_one(_norm):
                    orig_shape = _norm.get_shape().as_list()
                    _norm = tf.case(fn_pred_fn_pairs(order_rand[0], _norm), default=lambda: default_fn(_norm))
                    _norm = tf.case(fn_pred_fn_pairs(order_rand[1], _norm), default=lambda: default_fn(_norm))
                    _norm = tf.case(fn_pred_fn_pairs(order_rand[2], _norm), default=lambda: default_fn(_norm))
                    _norm.set_shape(orig_shape)

                    return _norm

                norm = tf.map_fn(_color_jitter_one, norm)

            if sub_mean == 1:
                IMAGENET_MEAN = tf.constant(np.load(mean_path).swapaxes(0, 1).swapaxes(1, 2)[:, :, ::-1],
                                            dtype=tf.float32)
                orig_dtype = tf.float32
                norm = norm - IMAGENET_MEAN

            if self.group == 'train':
                if self.withflip == 1 or with_flip == 1:
                    def _postprocess_flip(im):
                        # Original way of flipping, changing to random_uniform way to be more controllable
                        # im = tf.image.random_flip_left_right(im, seed = seed_random)
                        # return im
                        do_flip = tf.random_uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32,
                                                    seed=seed_random)

                        def __left_right_flip(im):
                            flipped = tf.image.flip_left_right(im)
                            if is_normal == 1:
                                # flipped = 256 - flipped
                                flipped_x, flipped_y, flipped_z = tf.unstack(flipped, axis=2)
                                flipped = tf.stack([256 - flipped_x, flipped_y, flipped_z], axis=2)
                            return flipped

                        return tf.cond(tf.less(do_flip[0], 0.5), fn1=lambda: __left_right_flip(im), fn2=lambda: im)

                    norm = tf.map_fn(_postprocess_flip, norm, dtype=norm.dtype)

                if with_noise == 1:
                    def _postprocess_noise(im):
                        do_noise = tf.random_uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32, seed=None)

                        def __add_noise(im):
                            curr_level = tf.random_uniform(shape=[1], minval=0, maxval=noise_level,
                                                           dtype=tf.float32,
                                                           seed=None)
                            curr_noise = tf.random_normal(shape=tf.shape(im), mean=0.0, stddev=curr_level,
                                                          dtype=tf.float32)

                            return tf.add(im, curr_noise)

                        # return tf.cond(tf.less(do_noise[0], 0.5), true_fn = lambda: __add_noise(im), false_fn = lambda: im)
                        return tf.cond(tf.less(do_noise[0], 0.5), fn1=lambda: __add_noise(im), fn2=lambda: im)

                    norm = tf.map_fn(_postprocess_noise, norm, dtype=norm.dtype)
            crop_images = tf.cast(norm, orig_dtype)
        else:
            if with_color_noise == 1 and self.group == 'train':
                order_temp = tf.constant([0, 1, 2], dtype=tf.int32)
                order_rand = tf.random_shuffle(order_temp, seed=seed_random)

                fn_pred_fn_pairs = lambda x, image: [
                    (tf.equal(x, order_temp[0]),
                     lambda: tf.image.random_saturation(image, 0.6, 1.4, seed=seed_random)),
                    (tf.equal(x, order_temp[1]), lambda: tf.image.random_brightness(image, 0.4, seed=seed_random)),
                ]
                default_fn = lambda image: tf.image.random_contrast(image, 0.6, 1.4, seed=seed_random)

                def _color_jitter_one(_norm):
                    orig_shape = _norm.get_shape().as_list()
                    _norm = tf.case(fn_pred_fn_pairs(order_rand[0], _norm), default=lambda: default_fn(_norm))
                    _norm = tf.case(fn_pred_fn_pairs(order_rand[1], _norm), default=lambda: default_fn(_norm))
                    _norm = tf.case(fn_pred_fn_pairs(order_rand[2], _norm), default=lambda: default_fn(_norm))
                    _norm.set_shape(orig_shape)

                    return _norm

                norm = tf.map_fn(_color_jitter_one, norm)

            if sub_mean == 1:
                IMAGENET_MEAN = tf.constant(np.load(mean_path).swapaxes(0, 1).swapaxes(1, 2)[:, :, ::-1],
                                            dtype=tf.float32)
                orig_dtype = tf.float32
                norm = norm - IMAGENET_MEAN
            if self.group == 'train':

                if self.size_vary_prep == 0 and size_vary_prep == 0:
                    shape_tensor = norm.get_shape().as_list()
                    if self.crop_each == 0:
                        crop_images = tf.random_crop(norm, [curr_batch_size, self.crop_size, self.crop_size,
                                                            shape_tensor[3]], seed=seed_random)
                    else:
                        # original implementation is not useful, deleted, see the end of this file
                        crop_images = tf.random_crop(norm, [curr_batch_size, self.crop_size, self.crop_size,
                                                            shape_tensor[3]], seed=seed_random)
                else:  # self.size_vary_prep==1
                    if shape_undefined == 0:
                        channel_num = norm.get_shape().as_list()[-1]
                    else:
                        channel_num = 3
                    RandomSizedCrop_with_para = lambda image: RandomSizedCrop(
                        image=image,
                        out_height=self.crop_size,
                        out_width=self.crop_size,
                        seed_random=seed_random,
                        channel_num=channel_num,
                        fix_asp_ratio=self.fix_asp_ratio,
                        size_minval=size_minval,
                    )
                    if shape_undefined == 0:
                        crop_images = tf.map_fn(RandomSizedCrop_with_para, norm)
                        curr_shape = crop_images.get_shape().as_list()
                        crop_images.set_shape([curr_batch_size] + curr_shape[1:])
                    else:
                        crop_images = RandomSizedCrop_with_para(norm)
                        crop_images = tf.expand_dims(crop_images, axis=0)

                if self.withflip == 1 or with_flip == 1:
                    def _postprocess_flip(im):
                        # Original way of flipping, changing to random_uniform way to be more controllable
                        # im = tf.image.random_flip_left_right(im, seed = seed_random)
                        # return im
                        do_flip = tf.random_uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32,
                                                    seed=seed_random)

                        def __left_right_flip(im):
                            flipped = tf.image.flip_left_right(im)
                            if is_normal == 1:
                                # flipped = 256 - flipped
                                flipped_x, flipped_y, flipped_z = tf.unstack(flipped, axis=2)
                                flipped = tf.stack([256 - flipped_x, flipped_y, flipped_z], axis=2)
                            return flipped

                        return tf.cond(tf.less(do_flip[0], 0.5), fn1=lambda: __left_right_flip(im), fn2=lambda: im)

                    crop_images = tf.map_fn(_postprocess_flip, crop_images, dtype=crop_images.dtype)

                if with_noise == 1:
                    def _postprocess_noise(im):
                        do_noise = tf.random_uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32, seed=None)

                        def __add_noise(im):
                            curr_level = tf.random_uniform(shape=[1], minval=0, maxval=noise_level,
                                                           dtype=tf.float32, seed=None)
                            curr_noise = tf.random_normal(shape=tf.shape(im), mean=0.0, stddev=curr_level,
                                                          dtype=tf.float32)

                            return tf.add(im, curr_noise)

                        # return tf.cond(tf.less(do_noise[0], 0.5), true_fn = lambda: __add_noise(im), false_fn = lambda: im)
                        return tf.cond(tf.less(do_noise[0], 0.5), fn1=lambda: __add_noise(im), fn2=lambda: im)

                    crop_images = tf.map_fn(_postprocess_noise, crop_images, dtype=crop_images.dtype)

            else:  # not self.group=='train'

                if shape_undefined == 0:
                    off = np.zeros(shape=[curr_batch_size, 4])
                    off[:, 0] = int((NOW_SIZE1 - self.crop_size) / 2)
                    off[:, 1] = int((NOW_SIZE2 - self.crop_size) / 2)
                    off[:, 2:4] = off[:, :2] + self.crop_size
                    off[:, 0] = off[:, 0] * 1.0 / (NOW_SIZE1 - 1)
                    off[:, 2] = off[:, 2] * 1.0 / (NOW_SIZE1 - 1)

                    off[:, 1] = off[:, 1] * 1.0 / (NOW_SIZE2 - 1)
                    off[:, 3] = off[:, 3] * 1.0 / (NOW_SIZE2 - 1)

                    box_ind = tf.constant(range(curr_batch_size))

                    crop_images = tf.image.crop_and_resize(norm, off, box_ind,
                                                           tf.constant([self.crop_size, self.crop_size]))
                else:
                    image = _aspect_preserving_resize(norm, 256)
                    image = _central_crop([image], self.crop_size, self.crop_size)[0]
                    image.set_shape([self.crop_size, self.crop_size, 3])
                    crop_images = image
                    crop_images = tf.expand_dims(crop_images, axis=0)

            crop_images = tf.cast(crop_images, orig_dtype)
        if curr_batch_size == 1:
            crop_images = tf.squeeze(crop_images, axis=[0])

        return crop_images

