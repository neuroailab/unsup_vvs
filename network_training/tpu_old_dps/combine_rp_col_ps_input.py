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

import os, random
import numpy as np

import tensorflow as tf
from utils import rgb_to_lab
import resnet_preprocessing

class Combine_RP_Color_PS_Input(object):
  """Generates ImageNet input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The fortmat of the data required is created by the script at:
      https://github.com/tensorflow/tpu-demos/blob/master/cloud_tpu/datasets/imagenet_to_gcs.py

  Args:
    is_training: `bool` for whether the input is for training
    data_dir: `str` for the directory of the training and validation data
    num_cores: `int` for the number of TPU cores
  """

  def __init__(self, is_training, pbr_dir, scene_dir, num_cores=8, num_grids=4, g_noise=0, down_sample=8, std=False):
    # self.image_preprocessing_fn = resnet_preprocessing.full_imagenet_augment
    self.image_preprocessing_fn = resnet_preprocessing.rp_preprocess_image
    self.rp_preprocessing_fn = resnet_preprocessing.rp_col_preprocess_image
    self.color_preprocessing_fn = resnet_preprocessing.col_preprocess_image
    self.is_training = is_training
    self.pbr_dir = pbr_dir
    self.scene_dir = scene_dir
    self.num_cores = num_cores
    self.down_sample = down_sample
    self.num_grids = num_grids
    self.std = std
    if is_training == True:
        self.g_noise = g_noise
    else:
        self.g_noise = 0

  def dataset_parser_pbr(self, value):
        """Parse an ImageNet record from a serialized string Tensor."""
        keys_to_features = {
            'mlt': tf.FixedLenFeature((), tf.string, '')
            }

        parsed = tf.parse_single_example(value, keys_to_features)
        print("parsed example", parsed)

        original_image = tf.reshape(parsed['mlt'], shape=[])
        original_image = tf.image.decode_png(original_image, channels=3)
        original_image = tf.image.convert_image_dtype(original_image, dtype=tf.float32)
        original_image.set_shape((480, 640, 3))
        return original_image

  def dataset_parser_scene(self, value):
        """Parse an ImageNet record from a serialized string Tensor."""
        keys_to_features = {
            'photo': tf.FixedLenFeature((), tf.string, '')
            }

        parsed = tf.parse_single_example(value, keys_to_features)
        print("parsed example", parsed)

        original_image = tf.reshape(parsed['photo'], shape=[])
        original_image = tf.image.decode_png(original_image, channels=3)
        original_image = tf.image.convert_image_dtype(original_image, dtype=tf.float32)
        original_image.set_shape((240, 320, 3))
        
        return original_image

  def dataset_parser_together(self, value):
    rp_pbr_image = self.rp_preprocessing_fn(
        image=value['pbr'],
        is_training=self.is_training,
        down_sample=self.down_sample
        )
    #print("g_noise:", self.g_noise)
    rp_pbr_image = self.image_preprocessing_fn(
        image=rp_pbr_image,
        is_training=self.is_training,
        num_grids=self.num_grids,
        g_noise=self.g_noise,
        std=self.std,
        sub_mean=False,
    )
    rp_scene_image = self.rp_preprocessing_fn(
        image=value['scene'],
        is_training=self.is_training,
        down_sample=self.down_sample
        )
    #print("g_noise:", self.g_noise)
    rp_scene_image = self.image_preprocessing_fn(
        image=rp_scene_image,
        is_training=self.is_training,
        num_grids=self.num_grids,
        g_noise=self.g_noise,
        std=self.std,
        sub_mean=False,
    )
    rp_image = tf.concat([rp_pbr_image, rp_scene_image], 0)
    color_pbr_image, color_pbr_label = self.color_preprocessing_fn(
        image=value['pbr'],
        is_training=self.is_training,
        down_sample=8,
        col_knn=True,
        combine_rp=True,
        )
    color_scene_image, color_scene_label = self.color_preprocessing_fn(
        image=value['scene'],
        is_training=self.is_training,
        col_knn=True,
        combine_rp=True,
        )
    color_pbr_image = tf.expand_dims(color_pbr_image, 0)
    color_pbr_label = tf.expand_dims(color_pbr_label, 0)
    color_scene_image = tf.expand_dims(color_scene_image, 0)
    color_scene_label = tf.expand_dims(color_scene_label, 0)

    color_image = tf.concat([color_pbr_image, color_scene_image], 0)
    color_label = tf.concat([color_pbr_label, color_scene_label], 0)
    return rp_image, color_image, color_label

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

    if self.is_training:
        pbr_file_pattern = os.path.join(self.pbr_dir, 'tfrecords', 'mlt', 'pbrnet_*')
        scene_file_pattern = os.path.join(self.scene_dir, 'scenenet_new', 'photo', 'data_*')
    else:
        pbr_file_pattern = os.path.join(self.pbr_dir, 'tfrecords_val', 'mlt', 'pbrnet_*')    
        scene_file_pattern = os.path.join(self.scene_dir, 'scenenet_new_val', 'photo', 'data_*')

    pbr_files = tf.data.Dataset.list_files(pbr_file_pattern)
    scene_files = tf.data.Dataset.list_files(scene_file_pattern)

    if self.is_training:
        pbr_files = pbr_files.shuffle(buffer_size=1024).repeat()   # 1024 files in dataset
        scene_files = scene_files.shuffle(buffer_size=1024).repeat()

    def fetch_dataset(filename):
        buffer_size = 8 * 1024 * 1024     # 8 MiB per file
        dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
        return dataset

    pbr_files = pbr_files.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=self.num_cores, sloppy=True))
    pbr_files = pbr_files.shuffle(1024)

    scene_files = scene_files.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=self.num_cores, sloppy=True))
    scene_files = scene_files.shuffle(1024)
    
    dataset_parser_dict = {}

    dataset_parser_dict['pbr'] = pbr_files.map(
        self.dataset_parser_pbr,
        num_parallel_calls=64)
    dataset_parser_dict['scene'] = scene_files.map(
        self.dataset_parser_scene,
        num_parallel_calls=64)
    
    zip_dataset = tf.data.Dataset.zip(dataset_parser_dict)
    zip_dataset = zip_dataset.repeat()
    zip_dataset = zip_dataset.map(
        self.dataset_parser_together,
        num_parallel_calls=64)
    zip_dataset = zip_dataset.shuffle(buffer_size=1200, seed=4) 
    zip_dataset = zip_dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    zip_dataset = zip_dataset.prefetch(2)     # Prefetch overlaps in-feed with training
    rp_images, color_images, color_labels = zip_dataset.make_one_shot_iterator().get_next()
    print("rp image:", rp_images)
    print("color image:", color_images)
    print("color label:", color_labels)

    rp_images = tf.reshape(rp_images, [batch_size * 2, 4, 96, 96, 3])
    
    # sample this number grids per image
    all_images = {}
    all_images['rp'] = tf.reshape(rp_images, [batch_size * 2, 4, 96, 96, 3])
    all_images['col'] = tf.reshape(color_images, [batch_size * 2, 256, 256, 3])
    all_images['col_labels'] = tf.reshape(color_labels, [batch_size * 2, 32, 32, 313])

    labels = tf.constant([[0], [1], [2], [3], [4], [5], [6], [7]])
    return all_images, labels

