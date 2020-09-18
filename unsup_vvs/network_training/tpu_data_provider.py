"""Input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import tensorflow as tf

import pdb

import unsup_vvs.network_training.resnet_preprocessing as resnet_preprocessing
from unsup_vvs.network_training.utils import rgb_to_lab
import unsup_vvs.network_training.resnet_th_preprocessing as prep_util


# Useful util function
def fetch_dataset(filename):
    buffer_size = 8 * 1024 * 1024     # 8 MiB per file
    dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
    return dataset


class TPUCombineWorld(object):
    """
    According to the dataset configuration, 
    create each dataset and build the iterator for each one
    """
    def __init__(
            self,
            data_path,
            group='train',
            num_cores=8,
            cfg_dataset={},
            which_imagenet='full',
            file_shuffle=True,
            imgnt_w_idx=False,
            resnet_prep=False,
            resnet_prep_size=False,
            *args, **kwargs
            ):
        # Set parameters useful for other functions
        self.data_path = data_path
        self.is_training = group=='train'
        self.num_cores = num_cores
        self.cfg_dataset = cfg_dataset
        self.which_imagenet = which_imagenet
        self.file_shuffle = file_shuffle
        self.resnet_prep = resnet_prep
        self.resnet_prep_size = resnet_prep_size
        self.imgnt_w_idx = imgnt_w_idx

    def build_dataset(
            self,
            list_file_dataset,
            parser_func,
            file_bf_size=1024,
            rec_bf_size=51200,
            ):
        '''
        Starting from a list_file_dataset, 
        do the parser, 
        return a dataset with the actual items
        '''
        dataset = list_file_dataset
        # Function to fetch and parse
        def _fetch_and_parse(dataset):
            # Shuffle the file list dataset if needed
            if self.is_training and self.file_shuffle:
                dataset = dataset.apply(
                        tf.contrib.data.shuffle_and_repeat(
                            file_bf_size))
            else:
                dataset = dataset.repeat()
            # Fetch the tfrecords
            dataset = dataset.apply(
                    tf.contrib.data.parallel_interleave(
                        fetch_dataset, 
                        cycle_length=self.num_cores, sloppy=True))
            # Shuffle the tfrecords if needed
            if self.is_training:
                dataset = dataset.apply(
                        tf.contrib.data.shuffle_and_repeat(
                            rec_bf_size))
            dataset = dataset.map(
                    parser_func,
                    num_parallel_calls=64)
            return dataset

        # Repeat the file list if needed
        dataset = _fetch_and_parse(dataset)
        return dataset

    def __parse_ImageNet(self, value):
        # Parse the tfrecord
        keys_to_features = {
                'images': tf.FixedLenFeature((), tf.string, ''),
                'labels': tf.FixedLenFeature([], tf.int64, -1)
                }
        if self.imgnt_w_idx:
            keys_to_features['index'] = tf.FixedLenFeature([], tf.int64, -1)
        parsed = tf.parse_single_example(value, keys_to_features)
        return parsed

    def __prep_rp_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)

        image = resnet_preprocessing.rp_preprocess_image(
                image,
                is_training=self.is_training,
                sub_mean=False) 
        image = tf.cast(image, tf.uint8)
        if self.cfg_dataset.get('rp_imagenet_color_norm', 0)==1 \
                and self.is_training:
            image = tf.reshape(image, [-1] + image.get_shape().as_list()[2:])
            image = prep_util.ColorJitter(
                    image, as_batch=True)
        return image

    def dataset_parser_rp_ImageNet(self, value):
        parsed = self.__parse_ImageNet(value)
        # Preprocessing for image
        image = tf.reshape(parsed['images'], shape=[])
        image = self.__prep_rp_image(image)
        ret_dict = {'image_rp_imagenet': image}
        return ret_dict

    def dataset_parser_col_ImageNet(self, value):
        parsed = self.__parse_ImageNet(value)
        # Preprocessing for image
        image = tf.reshape(parsed['images'], shape=[])
        l_image, Q_label = prep_util.col_image_prep(image, self.is_training)
        ret_dict = {
                'image_col_imagenet': l_image,
                'q_label_col_imagenet': Q_label}
        return ret_dict

    def dataset_parser_ImageNet_rp_embed(self, value):
        parsed = self.__parse_ImageNet(value)
        # Preprocessing for image
        image = tf.reshape(parsed['images'], shape=[])
        rp_image = self.__prep_rp_image(image)
        ret_dict = self.__get_imagenet_ret_dict(parsed)
        ret_dict['image_rp_imagenet'] = rp_image
        return ret_dict

    def __prep_imagenet_image(self, image):
        crop_size = 224
        size_minval = 0.2
        if self.resnet_prep_size:
            size_minval = 0.08
        image = prep_util.preprocessing_inst(
                image,
                crop_size,
                crop_size,
                self.is_training,
                size_minval=size_minval,
                skip_color=self.resnet_prep,
                skip_gray=self.resnet_prep,
                )
        image = tf.cast(image, tf.uint8)
        return image

    def __get_imagenet_ret_dict(self, parsed, name_suffix=''):
        # Preprocessing for image
        image = tf.reshape(parsed['images'], shape=[])
        image = self.__prep_imagenet_image(image)

        # Preprocessing for label 
        ret_dict = {}
        label = tf.cast(
            tf.reshape(parsed['labels'], shape=[]), 
            dtype=tf.int32)

        image_key_name = 'image_imagenet' + name_suffix
        label_key_name = 'label_imagenet' + name_suffix
        ret_dict = {image_key_name: image, label_key_name: label}

        # Preprocessing for index
        if self.imgnt_w_idx:
            index = tf.reshape(parsed['index'], shape=[])
            index_key_name = 'index_imagenet' + name_suffix
            ret_dict[index_key_name] = index
        return ret_dict

    def dataset_parser_ImageNet(self, value, name_suffix=''):
        """Parse an ImageNet record from a serialized string Tensor."""
        parsed = self.__parse_ImageNet(value)
        ret_dict = self.__get_imagenet_ret_dict(parsed, name_suffix)
        return ret_dict

    def dataset_parser_PSNet(
            self, value, 
            dataset_name='pbrnet', i_h=480, i_w=640):
        """Parse an ImageNet record from a serialized string Tensor."""
        keys_to_features = {
            'mlt': tf.FixedLenFeature((), tf.string, ''),
            'depth': tf.FixedLenFeature((), tf.string, '')
        }

        parsed = tf.parse_single_example(value, keys_to_features)
        original_image = tf.reshape(parsed['mlt'], shape=[])
        original_image = tf.image.decode_png(original_image, channels=3)
        original_image.set_shape((i_h, i_w, 3))
        depth_image = tf.reshape(parsed['depth'], shape=[])
        depth_image = tf.image.decode_png(depth_image, dtype=tf.uint16)
        depth_image.set_shape((i_h, i_w, 1))
        depth_image = tf.cast(depth_image, tf.int32)

        original_image, depth_image \
                = resnet_preprocessing.depth_image_preprocess_image(
                        original_image=original_image,
                        depth_image=depth_image,
                        is_training=self.is_training,
                        image_height=i_h,
                        image_width=i_w)
        depth_image = resnet_preprocessing.depth_preprocess_image(depth_image)
        ret_dict = {
                'image_%s' % dataset_name: original_image, 
                'depth_%s' % dataset_name: depth_image}
        return ret_dict

    def get_tfr_filenames(self, folder_name, file_pattern='*.tfrecords'):
        # Get list of tfrecord filenames for given folder_name 
        # fitting the given file_pattern
        tfrecord_pattern = os.path.join(folder_name, file_pattern)
        datasource = tf.gfile.Glob(tfrecord_pattern)
        datasource.sort()
        return datasource

    def _get_imgnt_list_dataset(self, curr_data_dir):
        file_pattern = 'train-*' if self.is_training else 'validation-*'
        list_file = self.get_tfr_filenames(curr_data_dir, file_pattern)
        file_bf_size = len(list_file)
        list_file_dataset = tf.data.Dataset.list_files(list_file)
        return list_file_dataset, file_bf_size

    def _add_imagenet_other_branch(self, dataset_prefix, inputs):
        all_input_keys = copy.deepcopy(inputs.keys())
        for each_key in all_input_keys:
            if each_key.endswith('_imagenet'):
                new_key = each_key.replace(
                        '_imagenet', 
                        '_%s' % dataset_prefix)
                inputs[new_key] = inputs[each_key]
        return inputs

    def get_input_dict(self, batch_size):
        '''
        This function will get inputs as dictionary
        '''
        dict_dataset = {}
        def _prefetch_and_batch(dataset):
            dataset = dataset.prefetch(batch_size)
            dataset = dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(batch_size))
            return dataset

        ## Build ImageNet with labels
        if self.cfg_dataset.get('imagenet', 0)==1: 
            # Build the list_file_dataset first
            curr_data_dir = self.data_path['imagenet/image_label_%s' \
                    % self.which_imagenet]
            list_file_dataset, file_bf_size \
                    = self._get_imgnt_list_dataset(curr_data_dir)
            # Build the actual dataset
            def _parser_func(value):
                if self.cfg_dataset.get('rp_embed_imagenet', 0)==0:
                    return self.dataset_parser_ImageNet(value)
                else:
                    return self.dataset_parser_ImageNet_rp_embed(value)
            dataset = self.build_dataset(
                    list_file_dataset, 
                    _parser_func,
                    file_bf_size=file_bf_size)
            dict_dataset['imagenet'] = dataset

        ## Build ImageNet for RP
        if self.cfg_dataset.get('rp_imagenet', 0)==1 \
                and self.cfg_dataset.get('rp_embed_imagenet', 0)==0:
            # Build the list_file_dataset first
            curr_data_dir = self.data_path['imagenet/image_label_%s' \
                    % self.which_imagenet]
            list_file_dataset, file_bf_size \
                    = self._get_imgnt_list_dataset(curr_data_dir)
            # Build the actual dataset
            def _parser_func(value):
                return self.dataset_parser_rp_ImageNet(value)
            dataset = self.build_dataset(
                    list_file_dataset, 
                    _parser_func,
                    file_bf_size=file_bf_size)
            dict_dataset['rp_imagenet'] = dataset

        ## Build ImageNet for Col
        if self.cfg_dataset.get('col_imagenet', 0)==1:
            # Build the list_file_dataset first
            curr_data_dir = self.data_path['imagenet/image_label_%s' \
                    % self.which_imagenet]
            list_file_dataset, file_bf_size \
                    = self._get_imgnt_list_dataset(curr_data_dir)
            # Build the actual dataset
            def _parser_func(value):
                return self.dataset_parser_col_ImageNet(value)
            dataset = self.build_dataset(
                    list_file_dataset, 
                    _parser_func,
                    file_bf_size=file_bf_size)
            dict_dataset['col_imagenet'] = dataset

        ## Build ImageNet without labels (for mean-teacher)
        if self.cfg_dataset.get('imagenet_un', 0)==1: 
            which_un = self.cfg_dataset.get('imagenet_un_which', 'full')
            # Build the list_file_dataset first
            curr_data_dir = self.data_path['imagenet/image_label_%s' % which_un]
            if self.imgnt_w_idx:
                which_un = self.cfg_dataset.get(
                        'imagenet_un_which', 'full_widx')
                assert 'widx' in which_un
                curr_data_dir \
                        = self.data_path['imagenet/image_label_%s' % which_un]

            # For Infant Imagenet
            if 'infant' in self.which_imagenet:
                if 'ctl' in self.which_imagenet:
                    print("Ctl Infant.")
                else:
                    print("Infant Unlabel Loaded.")
                    data_path_name = 'imagenet/image_label_infant'
                    if self.imgnt_w_idx:
                        data_path_name = 'imagenet/image_label_infant_widx'
                    curr_data_dir = self.data_path[data_path_name]

            list_file_dataset, file_bf_size \
                    = self._get_imgnt_list_dataset(curr_data_dir)
            # Build the actual dataset
            def _parser_func(value):
                return self.dataset_parser_ImageNet(value, name_suffix='_un')
            dataset = self.build_dataset(
                    list_file_dataset, 
                    _parser_func,
                    file_bf_size=file_bf_size)
            dict_dataset['imagenet_un'] = dataset

        if self.cfg_dataset.get('pbrnet', 0)==1: 
            curr_data_dir = self.data_path['pbrnet/depth_mlt']
            list_file_dataset, file_bf_size \
                    = self._get_imgnt_list_dataset(curr_data_dir)
            # Build the actual dataset
            def _parser_func(value):
                return self.dataset_parser_PSNet(value)
            dataset = self.build_dataset(
                    list_file_dataset, 
                    _parser_func,
                    file_bf_size=file_bf_size)
            dict_dataset['pbrnet'] = dataset

        if self.cfg_dataset.get('scenenet', 0)==1: 
            curr_data_dir = self.data_path['scenenet_new/depth_mlt']
            list_file_dataset, file_bf_size \
                    = self._get_imgnt_list_dataset(curr_data_dir)
            # Build the actual dataset
            def _parser_func(value):
                return self.dataset_parser_PSNet(
                        value, dataset_name='scenenet', i_h=240, i_w=320)
            dataset = self.build_dataset(
                    list_file_dataset, 
                    _parser_func,
                    file_bf_size=file_bf_size)
            dict_dataset['scenenet'] = dataset

        if self.cfg_dataset.get('saycam', 0)==1 \
                and self.cfg_dataset.get('saycam_all', 0)==0:
            curr_data_dir = self.data_path['saycam/frames']
            list_file_dataset, file_bf_size \
                    = self._get_imgnt_list_dataset(curr_data_dir)
            # Build the actual dataset
            def _parser_func(value):
                ret_dict = self.dataset_parser_ImageNet(value)
                new_ret_dict = {
                        'image_saycam': ret_dict['image_imagenet'][::-1, :, :],
                        'index_saycam': ret_dict['index_imagenet'],
                        'label_saycam': ret_dict['label_imagenet'],
                        }
                return new_ret_dict
            dataset = self.build_dataset(
                    list_file_dataset, 
                    _parser_func,
                    file_bf_size=file_bf_size)
            dict_dataset['saycam'] = dataset

        if self.cfg_dataset.get('saycam', 0)==1 \
                and self.cfg_dataset.get('saycam_all', 0) > 0:
            frm_each_it = self.cfg_dataset.get('saycam_all', 0)
            assert 125 // frm_each_it * frm_each_it == 125, \
                    "125 should be divisible by"\
                    + " the number of frames in each instance"
            curr_data_dir = self.data_path['saycam/all_frames']
            list_file_dataset, file_bf_size \
                    = self._get_imgnt_list_dataset(curr_data_dir)
            # Build the actual dataset
            def _parser_func(value):
                ret_dict = self.dataset_parser_ImageNet(value)
                old_idx = ret_dict['index_imagenet']
                new_idx = tf.cast(tf.floor(old_idx / frm_each_it), tf.int64)
                new_ret_dict = {
                        'image_saycam': ret_dict['image_imagenet'][::-1, :, :],
                        'index_saycam': new_idx,
                        'label_saycam': ret_dict['label_imagenet'],
                        }
                return new_ret_dict
            dataset = self.build_dataset(
                    list_file_dataset, 
                    _parser_func,
                    file_bf_size=file_bf_size)
            dict_dataset['saycam'] = dataset

        # Zip and batch datasets 
        dataset = tf.data.Dataset.zip(dict_dataset)
        dataset = _prefetch_and_batch(dataset)

        # Prefetch overlaps in-feed with training
        dataset = dataset.prefetch(2)     
        input_dict = dataset.make_one_shot_iterator().get_next()
        # Organize it to one big dict
        big_input_dict = {}
        for _,temp_d in input_dict.items():
            for _key,value in temp_d.items():
                big_input_dict[_key] = value
        # Add imagenet other branches
        if self.cfg_dataset.get('imagenet_branch2', 0)==1: 
            big_input_dict = self._add_imagenet_other_branch(
                    'imagenet_branch2', big_input_dict)
        return big_input_dict

    def input_fn(self, params):
        # Build each dataset
        ## Define function to prefetch and batch the dataset
        batch_size = params['batch_size']
        # Get inputs
        input_dict = self.get_input_dict(batch_size)

        # Organize it to two dicts
        images_dict = {}
        labels_dict = {}
        for _key, value in input_dict.items():
            if value.dtype==tf.uint8:
                value = tf.cast(value, tf.float32)
            if 'label' in _key:
                labels_dict[_key] = value
            else:
                images_dict[_key] = value
        # For now label needs to be one tensor
        return images_dict, list(labels_dict.values())[0]
