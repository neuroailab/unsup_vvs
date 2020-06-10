from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import cPickle

from tfutils import base, data, optimizer

import json
import copy
import argparse

import coco_provider
import pdb

sys.path.append('../no_tfutils/')
from vgg_preprocessing import preprocess_image
import resnet_th_preprocessing
from resnet_th_preprocessing import RandomSizedCrop, \
                ColorNormalize, _aspect_preserving_resize, _central_crop, \
                RandomSizedCrop_from_jpeg

class Combine_world:

    def __init__(self,
                 data_path,
                 group='train',
                 batch_size=1,
                 n_threads=4,
                 crop_size=None,
                 cfg_dataset={},
                 depthnormal=0,
                 depthnormal_div=None,
                 queue_params=None,
                 withflip=0, 
                 whichimagenet='full',
                 no_shuffle=0,
                 whichcoco=0,
                 crop_time=5,
                 crop_rate=5,
                 replace_folder=None,
                 with_color_noise=0,
                 which_place=0,
                 size_vary_prep=0,
                 fix_asp_ratio=0,
                 img_size_vary_prep=0,
                 sm_full_size=0, # sm_add
                 col_size=0,
                 prob_gray=None,
                 size_minval=0.08,
                 val_on_train=0,
                 *args, **kwargs
                 ):

        with tf.variable_scope('TPUReplicate'):
            with tf.variable_scope('loop'):
                tpu_1 = tf.Variable(tf.constant(0.01), dtype=tf.float32, name='beta1_power')
                tpu_2 = tf.Variable(tf.constant(0.01), dtype=tf.float32, name='beta2_power')

        self.group = group
        self.batch_size = batch_size
        self.queue_params = queue_params
        self.withflip = withflip
        self.size_vary_prep = size_vary_prep
        self.fix_asp_ratio = fix_asp_ratio
        self.col_size = col_size
        self.prob_gray = prob_gray
        self.size_minval = size_minval

        self.shuffle_flag = group=='train'

        if no_shuffle==1:
            self.shuffle_flag = False

        self.crop_size = 224
        if not crop_size==None:
            self.crop_size = crop_size

        self.all_providers = []

        if cfg_dataset.get('scenenet', 0)==1:
            # Keys for scenenet, (240, 320), raw
            self.image_scenenet = 'image_scenenet'
            self.depth_scenenet = 'depth_scenenet'
            self.normal_scenenet = 'normal_scenenet'
            self.instance_scenenet = 'instance_scenenet'

            postproc_scenenet = (self.postproc_flag, (), 
                    {'NOW_SIZE1':240, 
                    'NOW_SIZE2':320, 
                    'seed_random':0, 
                    'sm_full_size':sm_full_size})

            postprocess_scenenet = {self.image_scenenet: [
                (self.postprocess_images, (), 
                {'dtype_now':tf.uint8, 
                'shape_now':(240, 320, 3)}), 
                postproc_scenenet]}

            need_normal = cfg_dataset.get('scene_normal', 1)==1
            need_depth = cfg_dataset.get('scene_depth', 1)==1
            need_instance = cfg_dataset.get('scene_instance', 0)==1

            if need_normal:
                postproc_scenenet_normal = postproc_scenenet
                # Special flipping to normals
                postproc_scenenet_normal = (
                        self.postproc_flag, (), 
                        {'NOW_SIZE1':240, 'NOW_SIZE2':320, 
                        'seed_random':0, 'is_normal':1,})

                postprocess_scenenet[self.normal_scenenet] = [
                    (self.postprocess_images, (), 
                    {'dtype_now':tf.uint8, 'shape_now':(240, 320, 3)}), 
                    postproc_scenenet_normal]

            if need_depth:
                postprocess_scenenet[self.depth_scenenet] = [
                    (self.postprocess_images, (), 
                    {'dtype_now':tf.uint16, 'shape_now':(240, 320, 1)}), 
                    postproc_scenenet]

                if depthnormal==1:
                    postprocess_scenenet[self.depth_scenenet].append(
                        (self.postprocess_normalize, (), 
                        {'depthnormal_div':depthnormal_div}))

            if need_instance:
                postprocess_scenenet[self.instance_scenenet] = [
                    (self.postprocess_images, (), 
                    {'dtype_now' : tf.uint16, 'shape_now' : (240, 320, 1)}), 
                    postproc_scenenet]

            source_prefix = 'scenenet_new'

            source_dirs_scenenet = [data_path["%s/%s/images" % (source_prefix, group)]]
            if need_normal:
                source_dirs_scenenet.append(data_path["%s/%s/normals" % (source_prefix, group)])
            if need_depth:
                source_dirs_scenenet.append(data_path["%s/%s/depths" % (source_prefix, group)])
            if need_instance:
                source_dirs_scenenet.append(data_path["%s/%s/instances" % (source_prefix, group)])

            trans_dicts_scenenet = [{'photo': self.image_scenenet}]

            if need_normal:
                trans_dicts_scenenet.append({'normals': self.normal_scenenet})
            if need_depth:
                trans_dicts_scenenet.append({'depth': self.depth_scenenet})
            if need_instance:
                trans_dicts_scenenet.append({'classes': self.instance_scenenet})

            self.all_providers.append(data.TFRecordsParallelByFileProvider(source_dirs = source_dirs_scenenet, 
                                            trans_dicts = trans_dicts_scenenet, 
                                            postprocess = postprocess_scenenet, 
                                            batch_size = batch_size, 
                                            n_threads=n_threads,
                                            shuffle = self.shuffle_flag,
                                            *args, **kwargs
                                            ))

        if cfg_dataset.get('pbrnet', 0)==1:
            # Keys for pbrnet, (480, 640), png, TODO: valid?
            self.image_pbrnet = 'image_pbrnet'
            self.depth_pbrnet = 'depth_pbrnet'
            self.normal_pbrnet = 'normal_pbrnet'
            self.instance_pbrnet = 'instance_pbrnet'

            postproc_pbrnet = (self.postproc_flag, (), 
                    {'NOW_SIZE1':480, 
                    'NOW_SIZE2':640, 
                    'seed_random':2, 
                    'sm_full_size':sm_full_size})
            postprocess_pbrnet = {self.image_pbrnet: [(self.postprocess_images, (), 
                {'dtype_now':tf.uint8, 'shape_now':(480, 640, 3)}), 
                postproc_pbrnet]}

            need_normal = cfg_dataset.get('pbr_normal', 1)==1
            need_depth = cfg_dataset.get('pbr_depth', 1)==1
            need_instance = cfg_dataset.get('pbr_instance', 0)==1

            if need_normal:
                postproc_pbrnet_normal = postproc_pbrnet
                # Do some special flipping to normals
                postproc_pbrnet_normal = (self.postproc_flag, (), 
                        {'NOW_SIZE1':480, 
                        'NOW_SIZE2':640, 
                        'seed_random':2, 
                        'is_normal':1})
                postprocess_pbrnet[self.normal_pbrnet] = [
                    (self.postprocess_images, (), 
                    {'dtype_now' : tf.uint8, 'shape_now' : (480, 640, 3)}), 
                    postproc_pbrnet_normal]
            if need_depth:
                postprocess_pbrnet[self.depth_pbrnet] = [
                    (self.postprocess_images, (), 
                    {'dtype_now':tf.uint16, 'shape_now':(480, 640, 1)}), 
                    postproc_pbrnet]
                if depthnormal==1:
                    postprocess_pbrnet[self.depth_pbrnet].append(
                        (self.postprocess_normalize, (), 
                        {'depthnormal_div':depthnormal_div}))

            if need_instance:
                postprocess_pbrnet[self.instance_pbrnet] = [
                    (self.postprocess_images, (), 
                    {'dtype_now':tf.uint8, 'shape_now':(480, 640, 1)}), 
                    postproc_pbrnet]

            source_dirs_pbrnet = [data_path["pbrnet/%s/images" % group]]
            if need_normal:
                source_dirs_pbrnet.append(data_path["pbrnet/%s/normals" % group])
            if need_depth:
                source_dirs_pbrnet.append(data_path["pbrnet/%s/depths" % group])
            if need_instance:
                source_dirs_pbrnet.append(data_path["pbrnet/%s/instances" % group])

            trans_dicts_pbrnet = [{'mlt': self.image_pbrnet}]
            if need_normal:
                trans_dicts_pbrnet.append({'normal': self.normal_pbrnet})
            if need_depth:
                trans_dicts_pbrnet.append({'depth': self.depth_pbrnet})
            if need_instance:
                trans_dicts_pbrnet.append({'category': self.instance_pbrnet})

            self.all_providers.append(
                    data.TFRecordsParallelByFileProvider(
                        source_dirs = source_dirs_pbrnet, 
                        trans_dicts = trans_dicts_pbrnet, 
                        postprocess = postprocess_pbrnet, 
                        batch_size = batch_size, 
                        n_threads=n_threads,
                        shuffle = self.shuffle_flag,
                        *args, **kwargs
                        ))

        imagenet_postproc_bp = { 
            'NOW_SIZE1':256, 
            'NOW_SIZE2':256, 
            'seed_random':3, 
            'with_flip':self.withflip, 
            'with_color_noise':with_color_noise,
            'size_vary_prep':img_size_vary_prep,
            'shape_undefined':1,
            'use_jpg':True,
            }
        if cfg_dataset.get('imagenet', 0)==1 \
                or cfg_dataset.get('rp', 0)==1 \
                or cfg_dataset.get('colorization', 0)==1: 
            if whichimagenet=='full_widx' and group=='val':
                whichimagenet = 'full'

            # 5 means full dataset, 6 means part, 20 means full with indexes
            # 40 means new imagenet (subsampled)
            # Use full dataset, in original shape
            self.image_imagenet = 'image_imagenet'
            self.label_imagenet = 'label_imagenet'
            with_index = whichimagenet=='full_widx'
            if with_index:
                self.index_imagenet = 'index_imagenet'

            assert batch_size==1, \
                    "For original size of imagenet, batch size must be 1!"
            postproc_imagenet_params = copy.deepcopy(imagenet_postproc_bp)
            postproc_imagenet = [
                    self.postproc_flag, (), 
                    postproc_imagenet_params]
            postprocess_imagenet = {
                    self.image_imagenet: [
                        (tf.reshape, (), {'shape':[]}), 
                        #(tf.image.decode_image, (), {'channels':3}),
                        postproc_imagenet],
                    self.label_imagenet: [(self.postproc_label, (), {})]}
            if with_index:
                postprocess_imagenet[self.index_imagenet] = \
                        postprocess_imagenet[self.label_imagenet]

            source_dirs_imagenet = [data_path["imagenet/image_label_%s" \
                    % whichimagenet ]]

            trans_dicts_imagenet = [{'images': self.image_imagenet, 
                                     'labels': self.label_imagenet}]
            if with_index:
                trans_dicts_imagenet[0]['index'] = self.index_imagenet

            if group=='train' or val_on_train==1:
                file_pattern = 'train*'
            else:
                file_pattern = 'val*'
            curr_batch_size = batch_size

            self.all_providers.append(
                    data.TFRecordsParallelByFileProvider(
                        source_dirs = source_dirs_imagenet, 
                        trans_dicts = trans_dicts_imagenet, 
                        postprocess = postprocess_imagenet, 
                        batch_size = curr_batch_size, 
                        n_threads=n_threads,
                        file_pattern=file_pattern,
                        shuffle = self.shuffle_flag,
                        *args, **kwargs
                        ))

        # Data provider for unlabeled imagenet, for Mean teacher implementation
        # I will just use the original size ImageNet
        if cfg_dataset.get('imagenet_un', 0)==1:

            # Use full dataset, in original shape
            self.image_imagenet_un = 'image_imagenet_un'
            self.label_imagenet_un = 'label_imagenet_un'

            assert batch_size==1,\
                    "For original size of imagenet, batch size must be 1!"
            postproc_imagenet_un_params = copy.deepcopy(imagenet_postproc_bp)
            postproc_imagenet_un = [
                    self.postproc_flag, (), 
                    postproc_imagenet_un_params]

            postprocess_imagenet_un = {
                    self.image_imagenet_un: [
                        (tf.reshape, (), {'shape':[]}), 
                        postproc_imagenet_un],
                    self.label_imagenet_un: [(self.postproc_label, (), {})]}

            source_dirs_imagenet_un = [data_path["imagenet/image_label_full"]]
            trans_dicts_imagenet_un = [{'images': self.image_imagenet_un, 
                                        'labels': self.label_imagenet_un}]
            if group=='train':
                file_pattern = 'train*'
            else:
                file_pattern = 'val*'

            curr_batch_size = batch_size

            self.all_providers.append(
                    data.TFRecordsParallelByFileProvider(
                        source_dirs=source_dirs_imagenet_un, 
                        trans_dicts=trans_dicts_imagenet_un, 
                        postprocess=postprocess_imagenet_un, 
                        batch_size=curr_batch_size, 
                        n_threads=n_threads,
                        file_pattern=file_pattern,
                        shuffle=self.shuffle_flag,
                        *args, **kwargs
                        ))


        if cfg_dataset.get('coco', 0)==1:
            key_list = ['height', 'images', 'labels', 'num_objects', \
                    'segmentation_masks', 'width']
            BYTES_KEYs = ['images', 'labels', 'segmentation_masks']

            if whichcoco==0:
                source_dirs = [data_path['coco/%s/%s' % (self.group, v)] for v in key_list]
            else:
                source_dirs = [data_path['coco_no0/%s/%s' % (self.group, v)] for v in key_list]

            meta_dicts = [
                    {v : {'dtype': tf.string, 'shape': []}} if v in BYTES_KEYs else {v : {'dtype': tf.int64, 'shape': []}} 
                    for v in key_list]

            self.all_providers.append(
                    coco_provider.COCO(
                        source_dirs = source_dirs,
                        meta_dicts = meta_dicts,
                        group = group,
                        batch_size = batch_size,
                        n_threads = n_threads,
                        image_min_size = 240,
                        crop_height = 224,
                        crop_width = 224,
                        *args, **kwargs
                        ))

        if cfg_dataset.get('place', 0)==1:

            self.image_place = 'image_place'
            self.label_place = 'label_place'

            postproc_place = [self.postproc_flag, (), { 
                'NOW_SIZE1':256, 
                'NOW_SIZE2':256, 
                'seed_random':4, 
                'with_flip':self.withflip,
                'with_color_noise':with_color_noise,
                }]
            postprocess_place = {
                    self.image_place: [
                        (self.postprocess_images, (), {'dtype_now' : tf.uint8, 'shape_now' : (256, 256, 3)}), 
                        postproc_place],
                    self.label_place: [(self.postproc_label, (), {})]}

            if which_place==0:
                source_dirs_place = [
                        data_path["place/%s/images" % group], 
                        data_path["place/%s/labels" % group]
                        ]
            else:
                source_dirs_place = [
                        data_path["place/%s/images_part" % group], 
                        data_path["place/%s/labels_part" % group],
                        ]
            trans_dicts_place = [{'image': self.image_place}, {'label': self.label_place}]

            curr_batch_size = batch_size

            self.all_providers.append(
                    data.TFRecordsParallelByFileProvider(
                        source_dirs = source_dirs_place, 
                        trans_dicts = trans_dicts_place, 
                        postprocess = postprocess_place, 
                        batch_size = curr_batch_size,
                        n_threads=n_threads,
                        shuffle = self.shuffle_flag,
                        *args, **kwargs
                        ))

        if cfg_dataset.get('kinetics', 0)==1:
             
            # Not a good style, but there are some other requirements in this file
            import kinetics_provider

            #key_list = ['path', 'label_p', 'height_p', 'width_p']
            key_list = ['path', 'label_p']

            source_dirs = [data_path['kinetics/%s/%s' % (self.group, v)] for v in key_list]

            self.all_providers.append(
                    kinetics_provider.Kinetics(
                        source_dirs = source_dirs,
                        group = group,
                        batch_size = batch_size,
                        n_threads = n_threads,
                        crop_height = 224,
                        crop_width = 224,
                        crop_time = crop_time,
                        crop_rate = crop_rate,
                        replace_folder = replace_folder,
                        shuffle = self.shuffle_flag,
                        *args, **kwargs
                        ))

        if cfg_dataset.get('nyuv2', 0)==1:

            # Keys for nyuv2, (480, 640), png
            self.image_nyuv2 = 'image_nyuv2'
            self.depth_nyuv2 = 'depth_nyuv2'

            SIZE_1 = 480
            SIZE_2 = 640

            postproc_nyuv2 = [
                    self.postproc_flag, 
                    (), 
                    {'NOW_SIZE1':SIZE_1, 'NOW_SIZE2':SIZE_2, 'seed_random':5, 'sm_full_size':sm_full_size}]
            postprocess_nyuv2 = {
                    self.image_nyuv2: [
                        (self.postprocess_images, (), {'dtype_now' : tf.uint8, 'shape_now' : (SIZE_1, SIZE_2, 3)}), 
                        postproc_nyuv2], 
                    self.depth_nyuv2: [
                        (self.postprocess_images, (), {'dtype_now' : tf.uint16, 'shape_now' : (SIZE_1, SIZE_2, 1)}), 
                        postproc_nyuv2]}

            if depthnormal==1:
                postprocess_nyuv2[self.depth_nyuv2].append(
                        (self.postprocess_normalize, (), {'depthnormal_div':depthnormal_div}))

            source_prefix = 'nyuv2'
            source_dirs_nyuv2 = [
                    data_path["%s/%s/images" % (source_prefix, group)], 
                    data_path["%s/%s/depths" % (source_prefix, group)]]

            trans_dicts_nyuv2 = [{'image': self.image_nyuv2}, {'depth': self.depth_nyuv2}]
            self.all_providers.append(
                    data.TFRecordsParallelByFileProvider(
                        source_dirs = source_dirs_nyuv2, 
                        trans_dicts = trans_dicts_nyuv2, 
                        postprocess = postprocess_nyuv2, 
                        batch_size = batch_size, 
                        n_threads=n_threads,
                        shuffle = self.shuffle_flag,
                        *args, **kwargs
                        ))


    def postproc_label(self, labels):

        curr_batch_size = self.batch_size

        labels.set_shape([curr_batch_size])
        
        if curr_batch_size==1:
            labels = tf.squeeze(labels, axis = [0])

        return labels

    def postproc_flag(
            self, images, 
            NOW_SIZE1=256, 
            NOW_SIZE2=256, 
            seed_random=0, 
            curr_batch_size=None, 
            with_flip=0, 
            is_normal=0, 
            mean_path=None, 
            size_vary_prep=0,
            with_color_noise=0,
            shape_undefined=0,
            size_minval=None,
            sm_full_size=0, # sm_add
            use_jpg=False,
            ):
        if size_minval is None:
            size_minval = self.size_minval
        if curr_batch_size==None:
            curr_batch_size = self.batch_size
        size_vary_prep = size_vary_prep==1 or self.size_vary_prep==1

        if not use_jpg:
            orig_dtype = images.dtype
            norm = tf.cast(images, tf.float32)
        else:
            assert shape_undefined==1 and size_vary_prep, \
                    "Must vary the shape when use_jpg"
            norm = images
            orig_dtype = tf.uint8

        def _apply_flip(norm):
            def _postprocess_flip(im):
                do_flip = tf.random_uniform(
                        shape=[], minval=0, maxval=1, 
                        dtype=tf.float32, seed=seed_random+104)
                def __left_right_flip(im):
                    flipped = tf.image.flip_left_right(im)
                    if is_normal == 1:
                        flipped_x, flipped_y, flipped_z = \
                                tf.unstack(flipped, axis=2)
                        flipped = tf.stack(
                                [256 - flipped_x, flipped_y, flipped_z], 
                                axis=2)
                    return flipped
                return tf.cond(tf.less(do_flip, 0.5), 
                        lambda: __left_right_flip(im), 
                        lambda: im)
            norm = tf.map_fn(_postprocess_flip, norm, dtype=norm.dtype)
            return norm

        if sm_full_size==1:
            crop_images = tf.cast(norm, dtype=orig_dtype)
        else: # sm_full_size == 0
            if self.group=='train':
                if not size_vary_prep:
                    shape_tensor = norm.get_shape().as_list()
                    crop_images = tf.random_crop(
                            norm, 
                            [curr_batch_size, self.crop_size, \
                                    self.crop_size, shape_tensor[3]], 
                            seed=seed_random+106)
                else: # self.size_vary_prep==1
                    if shape_undefined==0:
                        channel_num = norm.get_shape().as_list()[-1]
                    else:
                        channel_num = 3
                    if not use_jpg:
                        crop_kwargs = {
                                'seed_random': seed_random+107,
                                'channel_num': channel_num,
                                'fix_asp_ratio': self.fix_asp_ratio,
                                }
                        crop_func = RandomSizedCrop
                    else:
                        crop_kwargs = {}
                        crop_func = RandomSizedCrop_from_jpeg
                    RandomSizedCrop_with_para = lambda image: crop_func(
                            image, 
                            out_height=self.crop_size,
                            out_width=self.crop_size,
                            size_minval=size_minval,
                            **crop_kwargs
                            )
                    if shape_undefined==0:
                        crop_images = tf.map_fn(
                                RandomSizedCrop_with_para, 
                                norm)
                        curr_shape = crop_images.get_shape().as_list()
                        crop_images.set_shape(
                                [curr_batch_size] + curr_shape[1:])
                    else:
                        print("go into the right function")
                        crop_images = RandomSizedCrop_with_para(norm)
                        crop_images = tf.expand_dims(crop_images, axis=0)

                if self.prob_gray is not None:
                    assert self.prob_gray>0 and self.prob_gray<1,\
                            "Prob_gray should be bewteen 0 and 1"
                    crop_images = resnet_th_preprocessing.ApplyGray(
                            crop_images, self.prob_gray, as_batch=True)
                if with_color_noise==1:
                    crop_images = resnet_th_preprocessing.ColorJitter(
                            crop_images, as_batch=True, 
                            shape_undefined=shape_undefined)
                if self.withflip==1 or with_flip==1:
                    crop_images = _apply_flip(crop_images)
            else: # not self.group=='train'
                if use_jpg:
                    norm = tf.image.decode_image(norm, channels=3)
                if shape_undefined==0:
                    off = np.zeros(shape = [curr_batch_size, 4])
                    off[:, 0] = int((NOW_SIZE1 - self.crop_size)/2)
                    off[:, 1] = int((NOW_SIZE2 - self.crop_size)/2)
                    off[:, 2:4] = off[:, :2] + self.crop_size
                    off[:, 0] = off[:, 0]*1.0/(NOW_SIZE1 - 1)
                    off[:, 2] = off[:, 2]*1.0/(NOW_SIZE1 - 1)

                    off[:, 1] = off[:, 1]*1.0/(NOW_SIZE2 - 1)
                    off[:, 3] = off[:, 3]*1.0/(NOW_SIZE2 - 1)

                    box_ind    = tf.constant(range(curr_batch_size))

                    crop_images = tf.image.crop_and_resize(
                            norm, off, box_ind, 
                            tf.constant([self.crop_size, self.crop_size]))
                else:
                    image = _aspect_preserving_resize(
                            norm, 
                            256 + self.col_size)
                    image = _central_crop(
                            [image], 
                            self.crop_size, self.crop_size)[0]
                    image.set_shape([self.crop_size, self.crop_size, 3])
                    crop_images = image
                    crop_images = tf.expand_dims(crop_images, axis=0)

            crop_images = tf.cast(
                    crop_images, dtype=orig_dtype)
        if curr_batch_size==1:
            crop_images = tf.squeeze(crop_images, axis=[0])

        return crop_images

    def postprocess_images(self, ims, dtype_now, shape_now):
        def _postprocess_images(im):
            im = tf.image.decode_png(im, dtype = dtype_now)
            im.set_shape(shape_now)
            if dtype_now==tf.uint16:
                im = tf.cast(im, tf.int32)
            return im
        if dtype_now==tf.uint16:
            write_dtype = tf.int32
        else:
            write_dtype = dtype_now
        return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=write_dtype)

    def postprocess_normalize(self, ims, depthnormal_div=None):
        def _postprocess_normalize_2(im):
            im = tf.cast(im, tf.float32)
            if not depthnormal_div is None:
                im = im/depthnormal_div

            im = tf.image.per_image_standardization(im)
            return im

        if self.batch_size==1:
            return _postprocess_normalize_2(ims)
        else:
            return tf.map_fn(lambda im: _postprocess_normalize_2(im), ims, dtype = tf.float32)
    
    def postprocess_resize(self, ims, newsize_1=240, newsize_2=320):
        return tf.image.resize_images(ims, (newsize_1, newsize_2))

    def init_ops(self):
        all_init_ops = [data_temp.init_ops() for data_temp in self.all_providers]
        num_threads = len(all_init_ops[0])

        self.ret_init_ops = []

        for indx_t in xrange(num_threads):
            curr_dict = {}
            for curr_init_ops in all_init_ops:
                curr_dict.update(curr_init_ops[indx_t])
            self.ret_init_ops.append(curr_dict)

        return self.ret_init_ops

