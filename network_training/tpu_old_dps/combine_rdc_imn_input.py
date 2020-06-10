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

MEAN_RGB = [0.485, 0.456, 0.406]

class Combine_RDC_ImageNet_Input(object):
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

    def __init__(self, is_training, pbr_dir, scene_dir, imn_dir, num_cores=8, ab_depth=False, down_sample=1, color_dp_tl=True, rp_dp_tl=False, combine_3_task=True):
        # self.image_preprocessing_fn = resnet_preprocessing.full_imagenet_augment
        self.image_preprocessing_fn = resnet_preprocessing.depth_image_preprocess_image
        self.depth_preprocessing_fn = resnet_preprocessing.depth_preprocess_image
        self.rp_preprocessing_fn = resnet_preprocessing.rp_preprocess_image
        self.color_preprocessing_fn = resnet_preprocessing.col_preprocess_image
        self.rp_col_preprocessing_fn = resnet_preprocessing.rp_col_preprocess_image

        self.is_training = is_training
        self.pbr_dir = pbr_dir
        self.scene_dir = scene_dir
        self.imn_dir = imn_dir
        self.num_cores = num_cores
        self.pbr_training_data_num = 366
        self.scene_training_data_num = 843
        self.random_seed = 3
        self.ab_depth = ab_depth
        self.down_sample = down_sample
        self.color_dp_tl = color_dp_tl
        self.rp_dp_tl = rp_dp_tl
        self.combine_3_task = combine_3_task

    def dataset_parser_pbr_mlt(self, value):
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
        if self.rp_dp_tl:
            offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
            original_image -= offset
        return original_image

    def dataset_parser_pbr_depth(self, value):
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

    def dataset_parser_scene_photo(self, value):
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
        if self.rp_dp_tl:
            offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
            original_image -= offset
        return original_image

    def dataset_parser_scene_depth(self, value):
        """Parse an ImageNet record from a serialized string Tensor."""
        keys_to_features = {
            'depth': tf.FixedLenFeature((), tf.string, '')
            }

        parsed = tf.parse_single_example(value, keys_to_features)
        print("parsed example", parsed)

        depth_image = tf.reshape(parsed['depth'], shape=[])
        depth_image = tf.image.decode_png(depth_image, dtype=tf.uint16)
        depth_image.set_shape((240, 320, 1))
        depth_image = tf.cast(depth_image, tf.int32)
        return depth_image
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
    
        image = self.rp_col_preprocessing_fn(
            image=image,
            is_training=self.is_training,
            down_sample=8,
            )
        #print("g_noise:", self.g_noise)
        image = self.rp_preprocessing_fn(
            image=image,
            is_training=self.is_training,
            num_grids=1,
            g_noise=0,
            std=False,
            sub_mean=False,
        )

        label = tf.cast(tf.reshape(parsed['labels'], shape=[]), dtype=tf.int32)

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
        l_image, Q_label = self.color_preprocessing_fn(
            image=image,
            is_training=self.is_training,
            down_sample=8,
            col_knn=True,
            combine_rp=True,
        )   

        print("data_provider output: image:", l_image)
        print("label:", Q_label)


        return l_image, Q_label


    def dataset_parser_together(self, value):
        print("together input:", value) 
        pbr_image, pbr_depth = self.image_preprocessing_fn(
                original_image = value['pbr_mlt'],
                depth_image = value['pbr_depth'],
                is_training = self.is_training,
                image_height=480,
                image_width=640,
                color_dp_tl=self.color_dp_tl,
                combine_3_task=self.combine_3_task,
                )
        scene_image, scene_depth = self.image_preprocessing_fn(
                original_image = value['scene_photo'],
                depth_image = value['scene_depth'],
                is_training = self.is_training,
                image_height=240,
                image_width=320,
                color_dp_tl=self.color_dp_tl,
                combine_3_task=self.combine_3_task,
                )    
        pbr_depth = self.depth_preprocessing_fn(
            image=pbr_depth,
            ab_depth=self.ab_depth,
        ) 
        scene_depth = self.depth_preprocessing_fn(
            image=scene_depth,
            ab_depth=self.ab_depth,
        ) 

        color_image, color_label = value['col_file']
        rp_image, _ = value['rp_file']

        return pbr_image, pbr_depth, scene_image, scene_depth, rp_image, color_image, color_label


    def input_fn(self, params):

        batch_size = params['batch_size']

        if self.is_training:
            pbr_mlt_file_pattern = os.path.join(self.pbr_dir, 'tfrecords', 'mlt', 'pbrnet_*')
            pbr_depth_file_pattern = os.path.join(self.pbr_dir, 'tfrecords', 'depth', 'pbrnet_*')
            scene_photo_file_pattern = os.path.join(self.scene_dir, 'scenenet_new', 'photo', 'data_*')
            scene_depth_file_pattern = os.path.join(self.scene_dir, 'scenenet_new', 'depth', 'data_*')
            rp_file_pattern = os.path.join(self.imn_dir, 'train-*')
        else:
            pbr_mlt_file_pattern = os.path.join(self.pbr_dir, 'tfrecords_val', 'mlt', 'pbrnet_*')    
            pbr_depth_file_pattern = os.path.join(self.pbr_dir, 'tfrecords_val', 'depth', 'pbrnet_*')
            scene_photo_file_pattern = os.path.join(self.scene_dir, 'scenenet_new_val', 'photo', 'data_*')
            scene_depth_file_pattern = os.path.join(self.scene_dir, 'scenenet_new_val', 'depth', 'data_*')
            rp_file_pattern = os.path.join(self.imn_dir, 'validation-*') 

        pbr_mlt_list = self.get_tfr_filenames(pbr_mlt_file_pattern) 
        pbr_mlt_files = tf.data.Dataset.list_files(pbr_mlt_list)

        pbr_depth_list = self.get_tfr_filenames(pbr_depth_file_pattern)
        pbr_depth_files = tf.data.Dataset.list_files(pbr_depth_list)

        scene_photo_list = self.get_tfr_filenames(scene_photo_file_pattern)
        scene_photo_files = tf.data.Dataset.list_files(scene_photo_list) 

        scene_depth_list = self.get_tfr_filenames(scene_depth_file_pattern)
        scene_depth_files = tf.data.Dataset.list_files(scene_depth_list)

        rp_files = tf.data.Dataset.list_files(rp_file_pattern)

        if self.is_training:
            pbr_mlt_files = pbr_mlt_files.repeat()
            pbr_depth_files = pbr_depth_files.repeat()
            scene_photo_files = scene_photo_files.repeat()
            scene_depth_files = scene_depth_files.repeat()
            rp_files = rp_files.shuffle(buffer_size=1024).repeat()
        else:
            pbr_mlt_files = pbr_mlt_files.repeat()
            pbr_depth_files = pbr_depth_files.repeat()
            scene_photo_files = scene_photo_files.repeat()
            scene_depth_files = scene_depth_files.repeat()
            rp_files = rp_files.repeat()

        files_dict = {'pbr_mlt': pbr_mlt_files, 'pbr_depth': pbr_depth_files, 'scene_photo': scene_photo_files, 'scene_depth': scene_depth_files, 'rp_file': rp_files}

        def fetch_dataset(filename):
            buffer_size = 8 * 1024 * 1024  # 1024 MiB per file
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset
        
        dataset_dict = {
                source: curr_dataset.apply(
                    tf.contrib.data.parallel_interleave(
                        fetch_dataset, cycle_length=self.num_cores, sloppy=False)) \
                   for source, curr_dataset in files_dict.items()
                }

        dataset_parser_dict = {}
        dataset_parser_dict['pbr_mlt'] = dataset_dict['pbr_mlt'].map(
            self.dataset_parser_pbr_mlt,
            num_parallel_calls=64)
        dataset_parser_dict['pbr_depth'] = dataset_dict['pbr_depth'].map(
            self.dataset_parser_pbr_depth,
            num_parallel_calls=64)
        dataset_parser_dict['scene_photo'] = dataset_dict['scene_photo'].map(
            self.dataset_parser_scene_photo,
            num_parallel_calls=64)
        dataset_parser_dict['scene_depth'] = dataset_dict['scene_depth'].map(
            self.dataset_parser_scene_depth,
            num_parallel_calls=64)
        dataset_parser_dict['rp_file'] = dataset_dict['rp_file'].map(
            self.dataset_parser_rp,
            num_parallel_calls=64)
        dataset_parser_dict['col_file'] = dataset_dict['rp_file'].shuffle(buffer_size=1024, seed=5).map(
            self.dataset_parser_col,
            num_parallel_calls=64)
       
        zip_dataset = tf.data.Dataset.zip(dataset_parser_dict)
        zip_dataset = zip_dataset.repeat()
        
        zip_dataset = zip_dataset.map(
                self.dataset_parser_together,
                num_parallel_calls=64)
        
        zip_dataset = zip_dataset.shuffle(buffer_size=1200, seed=4)

        zip_dataset = zip_dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))

        zip_dataset = zip_dataset.prefetch(2)  # Prefetch overlaps in-feed with training

        pbr_image, pbr_depth, scene_image, scene_depth, rp_image, color_image, color_label = zip_dataset.make_one_shot_iterator().get_next()

        original_images = tf.concat([pbr_image, scene_image], 0)
        depth_images = tf.concat([pbr_depth, scene_depth], 0)

        tmp_concat = tf.concat([original_images, depth_images], 3)

        tmp_concat = tf.random_shuffle(tmp_concat, seed=5)

        original_image = tmp_concat[:,:,:,:3]
        depth_image = tmp_concat[:,:,:,3:]

        print("depth image:", original_image)
        print("depth label:", depth_image)
        print("rp image:", rp_image)
        print("color image:", color_image)
        print("color label:", color_label)
        
        all_images = {}
        all_images['depth_image'] = original_image[:batch_size, :, :, :]
        all_images['depth_label'] = depth_image[:batch_size, :, :, :]
        all_images['rp_image'] = tf.reshape(rp_image, [batch_size, 4, 96, 96, 3])
        all_images['color_image'] = tf.reshape(color_image, [batch_size, 256, 256, 3])
        all_images['color_label'] = tf.reshape(color_label, [batch_size, 32, 32, 313])
        labels = tf.constant([[0], [1], [2], [3], [4], [5], [6], [7]])

        return all_images, labels

    def get_tfr_filenames(self, file_pattern='*.tfrecords'):
        datasource = tf.gfile.Glob(file_pattern)
        datasource.sort()

        return datasource   
