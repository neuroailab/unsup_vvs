from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf

import json
import copy
import argparse
import pdb

class NeuralNewDataTF(object):
    def __init__(
            self,
            data_path,
            shape_dict,
            dtype_dict=None,
            group='train',
            shuffle_seed=0,
            map_pcall_num=64,
            img_out_size=224,
            buffer_size=2560,
            img_crop_size=None,
            **kwargs):
        self.data_path = data_path # dict of key to folder
        self.shape_dict = shape_dict
        self.group = group
        self.is_train = group=='train'
        self.shuffle_seed = shuffle_seed
        self.map_pcall_num = map_pcall_num
        self.img_out_size = img_out_size
        self.buffer_size = buffer_size
        self.img_crop_size = img_crop_size

        if self.is_train:
            self.file_pattern = 'train*'
        else:
            self.file_pattern = 'test*'

        all_keys = data_path.keys()
        all_keys = tuple(sorted(all_keys))
        assert isinstance(data_path, dict) and isinstance(shape_dict, dict), \
                "Data_path and shape_dict should be dictionaries"
        assert all_keys == tuple(sorted(shape_dict.keys())),\
                "Data_path and shape_dict should have same keys"

        if dtype_dict is None:
            dtype_dict = {}
            for key in all_keys:
                dtype_dict[key] = tf.float32
        else:
            assert isinstance(dtype_dict, dict), \
                    "Dtype_dict should be dictionaries"
            assert all_keys == tuple(sorted(dtype_dict.keys())),\
                    "Data_path and dtype_dict should have same keys"
        self.dtype_dict = dtype_dict

    def get_tfr_filenames(self, folder_name, file_pattern='train*'):
        # Get list of tfrecord filenames for given folder
        tfrecord_pattern = os.path.join(folder_name, file_pattern)
        datasource = tf.gfile.Glob(tfrecord_pattern)
        datasource.sort()
        return datasource

    def _get_file_datasets(self):
        source_lists = {
                source: self.get_tfr_filenames(
                    folder, 
                    file_pattern=self.file_pattern) \
                for source, folder in self.data_path.items()}

        file_datasets = {
                source: tf.data.Dataset.list_files(curr_files, shuffle=False) \
                for source, curr_files in source_lists.items()}

        if self.is_train:
            # Shuffle file names using the same shuffle_seed
            file_buffer_size = len(list(source_lists.values())[0])
            file_datasets = {
                    source: curr_dataset.shuffle(
                        buffer_size=file_buffer_size, 
                        seed=self.shuffle_seed).repeat() \
                    for source, curr_dataset in file_datasets.items()}
        else:
            file_datasets = {
                    source: curr_dataset.repeat() \
                    for source, curr_dataset in file_datasets.items()}
        return file_datasets

    def _get_record_datasets(self, file_datasets):
        # Create dataset for both
        def _fetch_dataset(filename):
            buffer_size = 8 * 1024 * 1024     # 8 MiB per file
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset

        each_dataset = {
                source: curr_dataset.apply(
                    tf.contrib.data.parallel_interleave(
                        _fetch_dataset, 
                        cycle_length=8, 
                        sloppy=False)) \
                for source,curr_dataset in file_datasets.items()
                }
        return each_dataset

    def _postproc_each(self, str_loaded, source):
        keys_to_features = {source: tf.FixedLenFeature((), tf.string)} 
        parsed = tf.parse_single_example(str_loaded, keys_to_features)

        curr_data = parsed[source]
        curr_data = tf.decode_raw(curr_data, self.dtype_dict[source])
        curr_data = tf.reshape(curr_data, self.shape_dict[source])
        if source == 'V4_ave':
            curr_data = curr_data[:88]
        if source == 'images':
            curr_data = tf.image.resize_images(
                    curr_data, 
                    [self.img_out_size, self.img_out_size])
            if self.img_crop_size is not None:
                start_pos = int((self.img_out_size - self.img_crop_size) / 2)
                end_pos = start_pos + self.img_crop_size
                curr_data = curr_data[start_pos : end_pos, start_pos : end_pos]
        return curr_data

    def _postproc_and_zip(self, each_dataset):
        # Decode raw first before zip
        each_dataset = {
                source: curr_dataset.map(
                    lambda x: self._postproc_each(x, source),
                    num_parallel_calls=self.map_pcall_num,
                    ) \
                for source, curr_dataset in each_dataset.items()
                }

        # Zip, repeat, batch
        zip_dataset = tf.data.Dataset.zip(each_dataset)
        return zip_dataset

    def build_dataset(self):
        file_datasets = self._get_file_datasets()
        each_dataset = self._get_record_datasets(file_datasets)
        zip_dataset = self._postproc_and_zip(each_dataset)

        if self.is_train:
            zip_dataset = zip_dataset.shuffle(
                    buffer_size=self.buffer_size, 
                    seed=self.shuffle_seed,
                    )
        return zip_dataset

    # entry point for TFUtils
    def input_fn(self, batch_size, **kwargs):
        self.batch_size = batch_size
        zip_dataset = self.build_dataset()
        zip_dataset = zip_dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        zip_dataset = zip_dataset.prefetch(2)
        zip_iter = zip_dataset.make_one_shot_iterator()
        input_dict = zip_iter.get_next()
        return input_dict


def v4it_test():
    data_path = {
            'images': '/data5/chengxuz/Dataset/neural_resp/images/V4IT/tf_records/images',
            'V4_ave': '/data5/chengxuz/Dataset/neural_resp/neural_resp/V4IT/V4_ave',
            'IT_ave': '/data5/chengxuz/Dataset/neural_resp/neural_resp/V4IT/IT_ave',
            'conv13': '/data5/chengxuz/Dataset/v4it_temp_results/V4IT/conv13',
            }
    shape_dict = {
            'images': (256, 256, 3),
            'V4_ave': (128,),
            'IT_ave': (168,),
            'conv13': (14, 14, 512),
            }
    data_provider = NeuralNewDataTF(data_path, shape_dict)
    input_dict = data_provider.input_fn(64)
    return input_dict


def v1v2_test():
    data_path = {
            'images': '/mnt/fs4/chengxuz/v1v2_related/tfrs/split_1/images',
            'V1_ave': '/mnt/fs4/chengxuz/v1v2_related/tfrs/split_1/V1_ave',
            'V2_ave': '/mnt/fs4/chengxuz/v1v2_related/tfrs/split_1/V2_ave',
            }
    shape_dict = {
            'images': (320, 320, 3),
            'V1_ave': (102,),
            'V2_ave': (103,),
            }
    data_provider = NeuralNewDataTF(data_path, shape_dict, buffer_size=270)
    input_dict = data_provider.input_fn(64)
    return input_dict


if __name__=='__main__':
    #input_dict = v4it_test()
    input_dict = v1v2_test()

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.Session()
    one_input = sess.run(input_dict)
    pdb.set_trace()
