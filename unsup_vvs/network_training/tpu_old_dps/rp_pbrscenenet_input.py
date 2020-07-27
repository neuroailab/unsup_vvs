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

import pdb


class PBRSceneNetDepthMltInput(object):
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

    def __init__(self, is_training, pbr_dir, scene_dir, num_cores=8, num_grids=4, g_noise=0, std=False):
        # self.image_preprocessing_fn = resnet_preprocessing.full_imagenet_augment
        self.image_preprocessing_fn = resnet_preprocessing.rp_preprocess_image
        self.is_training = is_training
        self.pbr_dir = pbr_dir
        self.scene_dir = scene_dir
        self.num_cores = num_cores
        self.pbr_training_data_num = 365
        self.scene_training_data_num = 843
        self.num_grids = num_grids
        self.std = std
        if is_training == True:
            self.g_noise = g_noise
        else:
            self.g_noise = 0

    def dataset_parser(self, value):
        """Parse an ImageNet record from a serialized string Tensor."""
        keys_to_features = {
            'mlt': tf.FixedLenFeature((), tf.string, ''),
            #'depth': tf.FixedLenFeature((), tf.string, '')
        }

        parsed = tf.parse_single_example(value, keys_to_features)
        print("parsed example", parsed)

        original_image = tf.reshape(parsed['mlt'], shape=[])
        original_image = tf.image.decode_png(original_image, channels=3)

        original_image = tf.image.convert_image_dtype(original_image, dtype=tf.float32)

        original_image = self.image_preprocessing_fn(
            image=original_image,
            is_training=self.is_training,
            num_grids=self.num_grids,
            g_noise = self.g_noise,
            std=self.std,
        )

        #depth_image = tf.reshape(parsed['depth'], shape=[])
        #depth_image = tf.image.decode_png(depth_image, channels=3)
        #depth_image = tf.image.convert_image_dtype(depth_image, dtype=tf.float32)

        return original_image, original_image

    def dataset_parser_2(self, value):
        """Parse an ImageNet record from a serialized string Tensor."""
        keys_to_features = {
            'photo': tf.FixedLenFeature((), tf.string, ''),
            #'depth': tf.FixedLenFeature((), tf.string, '')
        }

        parsed = tf.parse_single_example(value, keys_to_features)
        print("parsed example", parsed)

        original_image = tf.reshape(parsed['photo'], shape=[])
        original_image = tf.image.decode_png(original_image, channels=3)

        original_image = tf.image.convert_image_dtype(original_image, dtype=tf.float32)

        original_image = self.image_preprocessing_fn(
            image=original_image,
            is_training=self.is_training,
            num_grids=self.num_grids,
            g_noise = self.g_noise,
            std=self.std
        )

        #depth_image = tf.reshape(parsed['depth'], shape=[])
        #depth_image = tf.image.decode_png(depth_image, channels=3)
        #depth_image = tf.image.convert_image_dtype(depth_image, dtype=tf.float32)

        return original_image, original_image

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
        print("pbr path:", self.pbr_dir)
        # Shuffle the filenames to ensure better randomization.
        if self.is_training:
            pbr_file_pattern = os.path.join(self.pbr_dir, 'tfrecords', 'mlt', 'pbrnet_*')
        else:
            pbr_file_pattern = os.path.join(self.pbr_dir, 'tfrecords_val', 'mlt', 'pbrnet_*')

        dataset = tf.data.Dataset.list_files(pbr_file_pattern)
        print("dataset", dataset)
        '''
        scene_file_pattern = os.path.join(
            self.scene_dir, 'pbr_*' if self.is_training else 'pbr_*')
        dataset_tmp = tf.data.Dataset.list_files(scene_file_pattern)
        print("scene path:", self.scene_dir)

        print("dataset_tmp", dataset_tmp)

        dataset.concatenate(dataset_tmp)

        print("dataset", dataset)'''


        if self.is_training:
            dataset = dataset.shuffle(buffer_size=self.pbr_training_data_num)  # 1024 files in dataset
            dataset = dataset.repeat()

        def fetch_dataset(filename):
            #buffer_size = 1024 * 1024 * 1024  # 1024 MiB per file
            buffer_size = 1024 * 1024 * 1024  # 1024 MiB per file
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset

        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                fetch_dataset, cycle_length=self.num_cores, sloppy=True))

        dataset = dataset.shuffle(self.pbr_training_data_num)

        print("before parsing", dataset)
        dataset = dataset.map(
            self.dataset_parser,
            num_parallel_calls=64)
        print("after parsing", dataset)
        
        # Scenenet input:

        print("scenenet path:", self.scene_dir)
        # Shuffle the filenames to ensure better randomization.
        if self.is_training:
            scene_file_pattern = os.path.join(self.scene_dir, 'scenenet_new', 'photo', 'data_*')
        else:
            scene_file_pattern = os.path.join(self.scene_dir, 'scenenet_new_val', 'photo', 'data_*')

        dataset_tmp = tf.data.Dataset.list_files(scene_file_pattern)
        '''
        scene_file_pattern = os.path.join(
            self.scene_dir, 'pbr_*' if self.is_training else 'pbr_*')
        dataset_tmp = tf.data.Dataset.list_files(scene_file_pattern)
        print("scene path:", self.scene_dir)

        print("dataset_tmp", dataset_tmp)

        dataset.concatenate(dataset_tmp)

        print("dataset", dataset)'''


        if self.is_training:
            dataset_tmp = dataset_tmp.shuffle(buffer_size=self.scene_training_data_num)  # 1024 files in dataset
            dataset_tmp = dataset_tmp.repeat()


        dataset_tmp = dataset_tmp.apply(
            tf.contrib.data.parallel_interleave(
                fetch_dataset, cycle_length=self.num_cores, sloppy=True))

        dataset_tmp = dataset_tmp.shuffle(self.scene_training_data_num)

        print("before parsing", dataset_tmp)
        dataset_tmp = dataset_tmp.map(
            self.dataset_parser_2,
            num_parallel_calls=64)
        print("after parsing", dataset_tmp)
        
        dataset.concatenate(dataset_tmp)

        dataset = dataset.prefetch(batch_size)

        # For training, batch as usual. When evaluating, prevent accidentally
        # evaluating the same image twice by dropping the final batch if it is less
        # than a full batch size. As long as this validation is done with
        # consistent batch size, exactly the same images will be used.
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))

        dataset = dataset.prefetch(2)  # Prefetch overlaps in-feed with training

        original_images, labels = dataset.make_one_shot_iterator().get_next()
        if self.is_training: 
            images = tf.reshape(original_images, [batch_size * self.num_grids, 4, 96, 96, 3])
        else:
            images = tf.reshape(original_images, [batch_size, 4, 96, 96, 3])


        #depth_images = []

        return images, labels



