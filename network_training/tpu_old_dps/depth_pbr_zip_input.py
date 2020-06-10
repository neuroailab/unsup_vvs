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


class PBRNetZipDepthInput(object):
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
        self.random_seed = 3

    def dataset_parser_mlt(self, value):
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

    def dataset_parser_depth(self, value):
        """Parse an ImageNet record from a serialized string Tensor."""
        keys_to_features = {
            'depth': tf.FixedLenFeature((), tf.string, '')
            }

        parsed = tf.parse_single_example(value, keys_to_features)
        print("parsed example", parsed)

        depth_image = tf.reshape(parsed['depth'], shape=[])
        depth_image = tf.image.decode_png(depth_image, dtype=tf.uint16)
        depth_image.set_shape((480, 640, 1))
        depth_image = tf.cast(depth_image, tf.int32)
        return depth_image

    def dataset_parser_mlt_depth(self, value):
        print("mlt_depth input:", value) 
        original_image, depth_image = self.image_preprocessing_fn(
                original_image = value['mlt'],
                depth_image = value['depth'],
                is_training = self.is_training,
                image_height=480,
                image_width=640,
                )
        print("mlt_depth output:", original_image, depth_image)

        depth_image = self.depth_preprocessing_fn(
            image=depth_image,
        )        
        return original_image, depth_image


    def input_fn(self, params):

        batch_size = params['batch_size']

        if self.is_training:
            mlt_file_pattern = os.path.join(self.pbr_dir, 'tfrecords', 'mlt', 'pbrnet_*')
        else:
            mlt_file_pattern = os.path.join(self.pbr_dir, 'tfrecords_val', 'mlt', 'pbrnet_*')
           
        mlt_list = self.get_tfr_filenames(mlt_file_pattern)
        mlt_files = tf.data.Dataset.list_files(mlt_list)

        if self.is_training:
            depth_file_pattern = os.path.join(self.pbr_dir, 'tfrecords', 'depth', 'pbrnet_*')
        else:
            depth_file_pattern = os.path.join(self.pbr_dir, 'tfrecords_val', 'depth', 'pbrnet_*')
        
        depth_list = self.get_tfr_filenames(depth_file_pattern)
        depth_files = tf.data.Dataset.list_files(depth_list)

        if self.is_training:
            mlt_files = mlt_files.repeat()
            depth_files = depth_files.repeat()
        else:
            mlt_files = mlt_files.repeat()
            depth_files = depth_files.repeat()

        files_dict = { 'mlt': mlt_files, 'depth': depth_files}

        def fetch_dataset(filename):
            buffer_size = 1024 * 1024 * 1024  # 1024 MiB per file
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset
        
        dataset_dict = {
                source: curr_dataset.apply(
                    tf.contrib.data.parallel_interleave(
                        fetch_dataset, cycle_length=self.num_cores, sloppy=False)) \
                   for source, curr_dataset in files_dict.items()
                }

        dataset_parser_dict = {}
        dataset_parser_dict['mlt'] = dataset_dict['mlt'].map(
            self.dataset_parser_mlt,
            num_parallel_calls=64)
        dataset_parser_dict['depth'] = dataset_dict['depth'].map(
            self.dataset_parser_depth,
            num_parallel_calls=64)
        
        zip_dataset = tf.data.Dataset.zip(dataset_parser_dict)
        zip_dataset = zip_dataset.repeat()
        
        zip_dataset = zip_dataset.map(
                self.dataset_parser_mlt_depth,
                num_parallel_calls=64)
        
        zip_dataset = zip_dataset.shuffle(buffer_size=2400, seed=self.random_seed)

        zip_dataset = zip_dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))

        zip_dataset = zip_dataset.prefetch(2)  # Prefetch overlaps in-feed with training

        original_images, depth_images = zip_dataset.make_one_shot_iterator().get_next()
        print("original image and depth type:")
        print(original_images)
        print(depth_images)
        #pdb.set_trace()
        return original_images, depth_images

    def get_tfr_filenames(self, file_pattern='*.tfrecords'):
        datasource = tf.gfile.Glob(file_pattern)
        datasource.sort()
        #print(datasource)
        #pdb.set_trace()

        return datasource  
