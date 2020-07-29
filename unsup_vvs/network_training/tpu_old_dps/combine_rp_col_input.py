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

class Combine_RP_Color_Input(object):
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

  def __init__(self, is_training, data_dir, num_cores=8, num_grids=4, g_noise=0, down_sample=8, std=False):
    # self.image_preprocessing_fn = resnet_preprocessing.full_imagenet_augment
    self.image_preprocessing_fn = resnet_preprocessing.rp_preprocess_image
    self.rp_preprocessing_fn = resnet_preprocessing.rp_col_preprocess_image
    self.color_preprocessing_fn = resnet_preprocessing.col_preprocess_image
    self.is_training = is_training
    self.data_dir = data_dir
    self.col_dir = data_dir
    self.num_cores = num_cores
    self.down_sample = down_sample
    self.num_grids = num_grids
    self.std = std
    if is_training == True:
        self.g_noise = g_noise
    else:
        self.g_noise = 0

  def dataset_parser_rp(self, value):
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {
      'images': tf.FixedLenFeature((), tf.string, ''),
      'labels': tf.FixedLenFeature([], tf.int64, -1)
      }

    parsed = tf.parse_single_example(value, keys_to_features)
    print("parsed example", parsed)

    image = tf.reshape(parsed['images'], shape=[])
    image = tf.image.decode_jpeg(image, channels=3)
    
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    image = self.rp_preprocessing_fn(
        image=image,
        is_training=self.is_training,
        down_sample=self.down_sample
        )
    #print("g_noise:", self.g_noise)
    image = self.image_preprocessing_fn(
        image=image,
        is_training=self.is_training,
        num_grids=self.num_grids,
        g_noise=self.g_noise,
        std=self.std,
        sub_mean=False,
    )

    label = tf.cast(
        tf.reshape(parsed['labels'], shape=[]), dtype=tf.int32)

    return image, label

  def dataset_parser_col(self, value):
    keys_to_features = {
      'images': tf.FixedLenFeature((), tf.string, ''),
      'labels': tf.FixedLenFeature([], tf.int64, -1)
      }

    parsed = tf.parse_single_example(value, keys_to_features)
    print("parsed example", parsed)

    image = tf.reshape(parsed['images'], shape=[])
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    print("image shape:", image)
    l_image, Q_label  = self.color_preprocessing_fn(
        image=image,
        is_training=self.is_training,
        down_sample=self.down_sample,
        col_knn=True,
        combine_rp=True,
    )   

    print("data_provider output: image:", l_image)
    print("label:", Q_label)


    return l_image, Q_label

  def dataset_parser_together(self, value):
    col_image, col_label = value['col']
    rp_image, _ = value['rp']
    return rp_image, col_image, col_label

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
    rp_file_pattern = os.path.join(
        self.data_dir, 'train-*' if self.is_training else 'validation-*')

    rp_dataset = tf.data.Dataset.list_files(rp_file_pattern)

    col_file_pattern = os.path.join(
        self.data_dir, 'train-*' if self.is_training else 'validation-*')

    col_dataset = tf.data.Dataset.list_files(col_file_pattern)
    
    if self.is_training:
      rp_dataset = rp_dataset.shuffle(buffer_size=1024)   # 1024 files in dataset
      rp_dataset = rp_dataset.repeat()
      col_dataset = col_dataset.shuffle(buffer_size=1024).repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024     # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    rp_dataset = rp_dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=self.num_cores, sloppy=True))
    rp_dataset = rp_dataset.shuffle(1024)

    col_dataset = col_dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=self.num_cores, sloppy=True))
    col_dataset = col_dataset.shuffle(1024)
    
    dataset_parser_dict = {}

    dataset_parser_dict['rp'] = rp_dataset.map(
        self.dataset_parser_rp,
        num_parallel_calls=64)
    dataset_parser_dict['col'] = col_dataset.map(
        self.dataset_parser_col,
        num_parallel_calls=64)
    
    zip_dataset = tf.data.Dataset.zip(dataset_parser_dict)
    zip_dataset = zip_dataset.repeat()
    zip_dataset = zip_dataset.map(
        self.dataset_parser_together,
        num_parallel_calls=64)
   
    zip_dataset = zip_dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    zip_dataset = zip_dataset.prefetch(2)     # Prefetch overlaps in-feed with training
    rp_images, col_images, col_labels = zip_dataset.make_one_shot_iterator().get_next()

    if self.is_training:
        rp_images = tf.reshape(rp_images, [batch_size * self.num_grids, 4, 96, 96, 3]) 
    else:
        rp_images = tf.reshape(rp_images, [batch_size, 4, 96, 96, 3]) 
    # sample this number grids per image
    all_images = {}
    all_images['rp'] = rp_images
    all_images['col'] = col_images
    all_images['col_labels'] = col_labels
    labels = tf.constant([[0], [1], [2], [3], [4], [5], [6], [7]])
    return all_images, labels

