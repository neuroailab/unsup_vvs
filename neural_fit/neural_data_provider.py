from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import cPickle

from tfutils import base, data, optimizer

import json
import copy
import argparse
import pdb

class neuralTF(data.TFRecordsParallelByFileProvider):
    def __init__(self,
         data_path,
         batch_size=1,
         group='train',
         all_keys='',
         PCA_dir='',
         **kwargs
       ):
        key_list = all_keys.replace(':','-').split(',') + ['IT_ave', 'V4_ave']
        source_dirs = [os.path.join(data_path, each_key) for each_key in key_list]
        #print(key_list)
        #pdb.set_trace()

        postprocess = {v:[] for v in key_list}
        postprocess['IT_ave'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
        postprocess['IT_ave'].insert(1, (tf.reshape, ([-1] + [168], ), {}))
        postprocess['IT_ave'].insert(2, (self.set_shape_resps, (), {})) 

        postprocess['V4_ave'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
        postprocess['V4_ave'].insert(1, (tf.reshape, ([-1] + [88], ), {}))
        postprocess['V4_ave'].insert(2, (self.set_shape_resps, (), {})) 

        for other_key in key_list:
            if len(postprocess[other_key])>0:
                continue
            print('Load shape data for %s' % other_key)
            PCA_path = os.path.join(PCA_dir, '%s.pkl' % other_key)
            temp_data = cPickle.load(open(PCA_path, 'r'))

            postprocess[other_key].insert(0, (tf.decode_raw, (tf.float64, ), {})) 
            postprocess[other_key].insert(1, (tf.reshape, (temp_data['new_shape'], ), {}))
            postprocess[other_key].insert(2, (self.set_shape_resps, (), {})) 
            postprocess[other_key].insert(3, (tf.cast, (tf.float32, ), {})) 
	    
        if group=='train':
            file_pattern = 'train*.tfrecords'
        elif group=='val':
            file_pattern = 'test*.tfrecords'
        elif group=='all':
            file_pattern = '*.tfrecords'

	super(neuralTF, self).__init__(
	    source_dirs,
	    postprocess=postprocess,
            batch_size=batch_size,
            file_pattern=file_pattern,
	    **kwargs
	)

    def set_shape_resps(self, resps):
        curr_shape = resps.get_shape().as_list()
        curr_shape[0] = self.batch_size
        resps.set_shape(curr_shape)
        if self.batch_size==1:
            resps = tf.squeeze(resps, axis = [0])
        #resps = tf.Print(resps, [resps, resps.shape], message='Resps')
        
        return resps

class hvmTF_10ms(data.TFRecordsParallelByFileProvider):
    def __init__(self,
         data_path,
         which_split='split_0',
         resize=224,
         batch_size=1,
         group='train',
         **kwargs
       ):

	if resize is None:
	    self.resize = 256
	else:
	    self.resize = resize
        source_dirs = [data_path['%s/images' % which_split], 
                data_path['%s/labels' % which_split]]

	postprocess = {'images': [], 'labels': []}
	postprocess['images'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
	postprocess['images'].insert(1, (tf.reshape, ([-1] + [256, 256, 3], ), {}))
	postprocess['images'].insert(2, (self.postproc_imgs, (), {})) 

	postprocess['labels'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
	postprocess['labels'].insert(1, (tf.reshape, ([-1] + [256, 26], ), {}))
        postprocess['labels'].insert(2, (self.set_shape_resps, (), {})) 
	    
        if group=='train':
            file_pattern = 'train*.tfrecords'
        elif group=='val':
            file_pattern = 'test*.tfrecords'
        elif group=='all':
            file_pattern = '*.tfrecords'

	super(hvmTF_10ms, self).__init__(
	    source_dirs,
	    postprocess=postprocess,
            batch_size=batch_size,
            file_pattern=file_pattern,
	    **kwargs
	)

    def postproc_imgs(self, ims):

        def _postprocess_images(im):
	    im = tf.image.resize_images(im, [self.resize, self.resize])
	    return im
        resize_imgs = tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)
        curr_shape = resize_imgs.get_shape().as_list()
        resize_imgs.set_shape([self.batch_size] + curr_shape[1:])
        if self.batch_size==1:
            resize_imgs = tf.squeeze(resize_imgs, axis = [0])
        
        return resize_imgs

    def set_shape_resps(self, resps):
        curr_shape = resps.get_shape().as_list()
        resps.set_shape([self.batch_size, curr_shape[1], curr_shape[2]])
        if self.batch_size==1:
            resps = tf.squeeze(resps, axis = [0])
        
        return resps

class hvmTF(data.TFRecordsParallelByFileProvider):

    def __init__(self,
         data_path,
         which_split='split_0',
         resize=224,
         batch_size=1,
         group='train',
         **kwargs
	):

	self.resize = resize
        source_dirs = [data_path['%s/images' % which_split], 
                data_path['%s/IT' % which_split], 
                data_path['%s/V4' % which_split]]

        postprocess = {'images': [], 'IT_ave': [], 'V4_ave': []}
        postprocess['images'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
        postprocess['images'].insert(1, (tf.reshape, ([-1] + [256, 256, 3], ), {}))
        postprocess['images'].insert(2, (self.postproc_imgs, (), {})) 

        postprocess['IT_ave'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
        postprocess['IT_ave'].insert(1, (tf.reshape, ([-1] + [168], ), {}))
        postprocess['IT_ave'].insert(2, (self.set_shape_resps, (), {})) 

        postprocess['V4_ave'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
        postprocess['V4_ave'].insert(1, (tf.reshape, ([-1] + [128], ), {}))
        postprocess['V4_ave'].insert(2, (self.subselect_v4, (), {})) 
        postprocess['V4_ave'].insert(3, (self.set_shape_resps, (), {})) 
        
        if group=='train':
            file_pattern = 'train*.tfrecords'
        elif group=='val':
            file_pattern = 'test*.tfrecords'
        elif group=='all':
            file_pattern = '*.tfrecords'

        super(hvmTF, self).__init__(
            source_dirs,
            postprocess=postprocess,
            batch_size=batch_size,
            file_pattern=file_pattern,
            **kwargs
        )

    def postproc_imgs(self, ims):

        def _postprocess_images(im):
	    im = tf.image.resize_images(im, [self.resize, self.resize])
	    return im
        resize_imgs = tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)
        curr_shape = resize_imgs.get_shape().as_list()
        resize_imgs.set_shape([self.batch_size] + curr_shape[1:])
        if self.batch_size==1:
            resize_imgs = tf.squeeze(resize_imgs, axis = [0])
        
        return resize_imgs
    
    def subselect_v4(self, v4_resps):
        def _postprocess_images(v4_resp):
            v4_resp = v4_resp[:88]
	    return v4_resp
        return tf.map_fn(lambda v4_resp: _postprocess_images(v4_resp), v4_resps, dtype=tf.float32)

    def set_shape_resps(self, resps):
        curr_shape = resps.get_shape().as_list()
        resps.set_shape([self.batch_size, curr_shape[1]])
        if self.batch_size==1:
            resps = tf.squeeze(resps, axis = [0])
        
        return resps


class deepmindTF(data.TFRecordsParallelByFileProvider):

    def __init__(self,
         data_path,
         which_split='split_0',
         resize=224,
         batch_size=1,
         group='train',
         **kwargs
	):
        print(data_path)
	self.resize = resize
        source_dirs = [
                data_path['%s/block2_1' % which_split], 
                data_path['%s/block2_2' % which_split],
                data_path['%s/block2_3' % which_split],
                data_path['%s/block2_4' % which_split],
                data_path['%s/block3_1' % which_split],
                data_path['%s/block3_2' % which_split],
                data_path['%s/block3_3' % which_split],
                data_path['%s/block3_4' % which_split],
                data_path['%s/IT' % which_split], 
                data_path['%s/V4' % which_split]]
        '''

        postprocess['images'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
        postprocess['images'].insert(1, (tf.reshape, ([-1] + [256, 256, 3], ), {}))
        postprocess['images'].insert(2, (self.postproc_imgs, (), {})) 
        '''
        postprocess = {'block2_1': [], 'block2_2': [], 'block2_3': [], 'block2_4': [], 'block3_1': [], 'block3_2': [], 'block3_3': [], 'block3_4': [], 'IT_ave': [], 'V4_ave': []}
        
        postprocess['V4_ave'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
        postprocess['V4_ave'].insert(1, (tf.reshape, ([-1] + [128], ), {}))
        postprocess['V4_ave'].insert(2, (self.subselect_v4, (), {})) 
        postprocess['V4_ave'].insert(3, (self.set_shape_resps, (), {})) 

        postprocess['block2_2'].insert(0, (tf.decode_raw, (tf.float32, ), {}))
        postprocess['block2_2'].insert(1, (tf.reshape, ([-1] + [28, 28, 512], ), {})) 
        postprocess['block2_2'].insert(2, (self.set_shape_resps_deepmind, (), {})) 

        #postprocess['images'].insert(1, (tf.reshape, ([-1] + [256, 256, 3], ), {}))
        postprocess['block2_3'].insert(0, (tf.decode_raw, (tf.float32, ), {}))
        postprocess['block2_3'].insert(1, (tf.reshape, ([-1] + [28, 28, 512], ), {})) 
        postprocess['block2_3'].insert(2, (self.set_shape_resps_deepmind, (), {})) 

        postprocess['block2_1'].insert(0, (tf.decode_raw, (tf.float32, ), {}))
        postprocess['block2_1'].insert(1, (tf.reshape, ([-1] + [28, 28, 512], ), {}))
        postprocess['block2_1'].insert(2, (self.set_shape_resps_deepmind, (), {})) 

        postprocess['block2_4'].insert(0, (tf.decode_raw, (tf.float32, ), {}))
        postprocess['block2_4'].insert(1, (tf.reshape, ([-1] + [14, 14, 512], ), {}))
        postprocess['block2_4'].insert(2, (self.set_shape_resps_deepmind, (), {})) 

        postprocess['block3_1'].insert(0, (tf.decode_raw, (tf.float32, ), {}))                
        postprocess['block3_1'].insert(1, (tf.reshape, ([-1] + [14, 14, 1024], ), {}))
        postprocess['block3_1'].insert(2, (self.set_shape_resps_deepmind, (), {})) 

        postprocess['block3_2'].insert(0, (tf.decode_raw, (tf.float32, ), {}))
        postprocess['block3_2'].insert(1, (tf.reshape, ([-1] + [14, 14, 1024], ), {}))
        postprocess['block3_2'].insert(2, (self.set_shape_resps_deepmind, (), {})) 

        postprocess['block3_3'].insert(0, (tf.decode_raw, (tf.float32, ), {}))
        postprocess['block3_3'].insert(1, (tf.reshape, ([-1] + [14, 14, 1024], ), {}))
        postprocess['block3_3'].insert(2, (self.set_shape_resps_deepmind, (), {})) 

        postprocess['block3_4'].insert(0, (tf.decode_raw, (tf.float32, ), {}))
        postprocess['block3_4'].insert(1, (tf.reshape, ([-1] + [14, 14, 1024], ), {}))
        postprocess['block3_4'].insert(2, (self.set_shape_resps_deepmind, (), {})) 


        postprocess['IT_ave'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
        postprocess['IT_ave'].insert(1, (tf.reshape, ([-1] + [168], ), {}))
        postprocess['IT_ave'].insert(2, (self.set_shape_resps, (), {})) 

        
        
        if group=='train':
            file_pattern = 'train*.tfrecords'
        elif group=='val':
            file_pattern = 'test*.tfrecords'
        elif group=='all':
            file_pattern = '*.tfrecords'

        super(deepmindTF, self).__init__(
            source_dirs,
            postprocess=postprocess,
            batch_size=batch_size,
            file_pattern=file_pattern,
            **kwargs
        )

    def postproc_imgs(self, ims):

        def _postprocess_images(im):
	    im = tf.image.resize_images(im, [self.resize, self.resize])
	    return im
        resize_imgs = tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)
        curr_shape = resize_imgs.get_shape().as_list()
        resize_imgs.set_shape([self.batch_size] + curr_shape[1:])
        if self.batch_size==1:
            resize_imgs = tf.squeeze(resize_imgs, axis = [0])
        
        return resize_imgs
    
    def subselect_v4(self, v4_resps):
        def _postprocess_images(v4_resp):
            v4_resp = v4_resp[:88]
	    return v4_resp
        return tf.map_fn(lambda v4_resp: _postprocess_images(v4_resp), v4_resps, dtype=tf.float32)

    def set_shape_resps(self, resps):
        curr_shape = resps.get_shape().as_list()
        #print(curr_shape)
        resps.set_shape([self.batch_size, curr_shape[1]])
        if self.batch_size==1:
            resps = tf.squeeze(resps, axis = [0])
        
        return resps

    def set_shape_resps_deepmind(self, resps):
        curr_shape = resps.get_shape().as_list()
        resps.set_shape([self.batch_size, curr_shape[1], curr_shape[2], curr_shape[3]])
        if self.batch_size==1:
            resps = tf.squeeze(resps, axis = [0])
        
        return resps
