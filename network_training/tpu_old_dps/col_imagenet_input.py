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
from utils import *

import tensorflow as tf

import resnet_preprocessing

class Col_ImageNetInput(object):
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

  def __init__(self, is_training, data_dir, num_cores=8, soft=True, down_sample=8, col_tl=False, col_knn=False, combine_rp=False):
    # self.image_preprocessing_fn = resnet_preprocessing.full_imagenet_augment
    self.image_preprocessing_fn = resnet_preprocessing.col_preprocess_image
    self.is_training = is_training
    self.data_dir = data_dir
    self.num_cores = num_cores
    self.soft = soft
    self.down_sample = down_sample
    self.col_knn = col_knn
    self.col_tl=col_tl
    self.combine_rp=combine_rp

  def dataset_parser(self, value):
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
    print("image shape:", image)
    l_image, Q_label  = self.image_preprocessing_fn(
        image=image,
        is_training=self.is_training,
        soft=self.soft,
        down_sample=self.down_sample,
        col_knn=self.col_knn,
        col_tl=self.col_tl,
        combine_rp=self.combine_rp,
    )   
    if self.col_tl:
        Q_label = tf.cast(
                tf.reshape(parsed['labels'], shape=[]), dtype=tf.int32)

    print("data_provider output: image:", l_image)
    print("label:", Q_label)


    return l_image, Q_label

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
    file_pattern = os.path.join(
        self.data_dir, 'train-*' if self.is_training else 'validation-*')
    dataset = tf.data.Dataset.list_files(file_pattern)
    print("dataset", dataset)
    
    if self.is_training:
      dataset = dataset.shuffle(buffer_size=1024)   # 1024 files in dataset
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024     # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=self.num_cores, sloppy=True))
    dataset = dataset.shuffle(1024)

    print("before parsing", dataset)
    dataset = dataset.map(
        self.dataset_parser,
        num_parallel_calls=64)
        #num_parallel_calls=64)
    print("after parsing", dataset)
    dataset = dataset.prefetch(batch_size)

    
    # For training, batch as usual. When evaluating, prevent accidentally
    # evaluating the same image twice by dropping the final batch if it is less
    # than a full batch size. As long as this validation is done with
    # consistent batch size, exactly the same images will be used.
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(2)     # Prefetch overlaps in-feed with training
    images, labels = dataset.make_one_shot_iterator().get_next()

    return images, labels

