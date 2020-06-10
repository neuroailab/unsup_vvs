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


class PBRNetDepthInput(object):
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

    def __init__(self, is_training, pbr_dir, num_cores=8):
        # self.image_preprocessing_fn = resnet_preprocessing.full_imagenet_augment
        self.image_preprocessing_fn = resnet_preprocessing.depth_image_preprocess_image
        self.depth_preprocessing_fn = resnet_preprocessing.depth_preprocess_image
        self.is_training = is_training
        self.pbr_dir = pbr_dir
        self.num_cores = num_cores
        self.training_data_num = 366

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
        original_image.set_shape((480, 640, 3))

        depth_image = tf.reshape(parsed['depth'], shape=[])
        depth_image = tf.image.decode_png(depth_image, dtype=tf.uint16)
        depth_image.set_shape((480, 640, 1))
        depth_image = tf.cast(depth_image, tf.int32)
        original_image, depth_image = self.image_preprocessing_fn(
            original_image=original_image,
            depth_image=depth_image,
            is_training=self.is_training,
            image_height=480,
            image_width=640,
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


        if self.is_training:
            dataset = dataset.shuffle(buffer_size=self.training_data_num)  # 1024 files in dataset
            dataset = dataset.repeat()
        else:
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


