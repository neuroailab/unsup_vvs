
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
"""ImageNet preprocessing for ResNet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np
import pdb
from tensorflow.python.ops import control_flow_ops

from unsup_vvs.network_training.models.rp_col_utils import rgb_to_lab, ab_to_Q


slim = tf.contrib.slim

IMAGE_SIZE = 224
COL_IMAGE_SIZE = 256
IMAGE_RESIZE = 224

MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]




def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: `Tensor` image of shape [height, width, channels].
    offset_height: `Tensor` indicating the height offset.
    offset_width: `Tensor` indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3), ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: `Tensor` of image (it will be converted to floats in [0, 1]).
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional `str` for name scope.
  Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox


def _random_crop(image, size):
  """Make a random crop of (`size` x `size`)."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  random_image, bbox = distorted_bounding_box_crop(
      image,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=100,
      scope=None)
  bad = _at_least_x_are_true(tf.shape(image), tf.shape(random_image), 3)

  image = tf.cond(
      bad, lambda: _center_crop(_do_scale(image, size), size),
      lambda: tf.image.resize_bicubic([random_image], [size, size])[0])
  return image


def _flip(image, random_seed=3):
  """Random horizontal image flip."""
  image = tf.image.random_flip_left_right(image, seed=random_seed)
  return image


def _at_least_x_are_true(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are true."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _do_scale(image, size):
  """Rescale the image by scaling the smaller spatial dimension to `size`."""
  shape = tf.cast(tf.shape(image), tf.float32)
  w_greater = tf.greater(shape[0], shape[1])
  shape = tf.cond(w_greater,
                  lambda: tf.cast([shape[0] / shape[1] * size, size], tf.int32),
                  lambda: tf.cast([size, shape[1] / shape[0] * size], tf.int32))

  return tf.image.resize_bicubic([image], shape)[0]


def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs


def _center_crop(image, size):
  """Crops to center of image with specified `size`."""
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  offset_height = ((image_height - size) + 1) / 2
  offset_width = ((image_width - size) + 1) / 2
  image = _crop(image, offset_height, offset_width, size, size)
  return image


def _normalize(image):
  """Normalize the image to zero mean and unit variance."""
  offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
  image -= offset

  scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
  image /= scale
  return image


def preprocess_for_train(image, std=True, depth_imn_tl=0, l_channel=0):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _random_crop(image, IMAGE_SIZE)
  #image = random_sized_crop(image, IMAGE_SIZE, IMAGE_SIZE)

  if depth_imn_tl == 0:
    if std:
        image = _normalize(image)
    else:
        offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
        image -= offset

  image = _flip(image)
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  if l_channel==1:
      lab_image = rgb_to_lab(image)
      l_image = lab_image[:, :, :1]
      l_image = l_image - 50
      image = tf.tile(l_image, [1, 1, 3])
  # resizing
  if IMAGE_RESIZE is not None:
    print("resizing to", IMAGE_RESIZE)
    image = tf.image.resize_images(image, [IMAGE_RESIZE, IMAGE_RESIZE], align_corners=True)
  
  return image


def preprocess_for_eval(image, std=True, depth_imn_tl=0, l_channel=0):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _do_scale(image, IMAGE_SIZE + 32)
  if depth_imn_tl == 0:
    if std:
        image = _normalize(image)
    else:
        offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
        image -= offset

  image = _center_crop(image, IMAGE_SIZE)
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  if l_channel==1:
      lab_image = rgb_to_lab(image)
      l_image = lab_image[:, :, :1]
      l_image = l_image - 50
      image = tf.tile(l_image, [1, 1, 3])
  # resizing
  if IMAGE_RESIZE is not None:  
    image = tf.image.resize_images(image, [IMAGE_RESIZE, IMAGE_RESIZE], align_corners=True)  

  return image


def preprocess_image(image, is_training=False, std=True, depth_imn_tl=0, l_channel=0):
  """Preprocesses the given image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.

  Returns:
    A preprocessed image `Tensor`.
  """
  if is_training:
    return preprocess_for_train(image, std=std, depth_imn_tl=depth_imn_tl, l_channel=l_channel)
  else:
    return preprocess_for_eval(image, std=std, depth_imn_tl=depth_imn_tl, l_channel=l_channel)


def full_imagenet_augment(image, is_training=False,
                          crop_size=224,
                          now_size_x=256,
                          now_size_y=256,
                          shape_undefined=True,
                          flip=True,
                          size_vary_prep=True,
                          color_norm=True,
                          random_seed=3):
  """Implements the full imagenet preprocessing

  Args:
    image: of unknown shape, dtype=tf.float32
    is_training: tells how to do random or deterministic preprocessing

  Returns:
    A preprocessed image 'Tensor'.
  """
  orig_dtype = image.dtype
  norm_im = image # can cast to tf.float32 here
  
  if is_training: # mode == TRAIN

    # random cropping
    if size_vary_prep:
      if shape_undefined:
        n_ch = 3
      else:
        n_ch = norm_im.get_shape().as_list()[-1]
      random_crop_func = lambda im: random_sized_crop(im,
                                                      out_height=crop_size,
                                                      out_width=crop_size,
                                                      random_seed=random_seed,
                                                      num_channels=n_ch,
                                                      fix_aspect_ratio=False,
                                                      size_minval=0.08)
      crop_image = random_crop_func(norm_im)

      # if shape_undefined:
      #   crop_images = random_crop_func(norm_ims)
      #   # crop_images = tf.expand_dims(crop_images, axis=0)
      # else:
      #   crop_images = tf.map_fn(random_crop_func, norm_ims)
      #   curr_shape = crop_images.get_shape().as_list()
      #               crop_images.set_shape([curr_batch_size] + curr_shape[1:])

    else: # same size for all ims
      shape = norm_im.get_shape().as_list()
      crop_image = tf.random_crop(norm_im, [crop_size, crop_size, shape[2]], seed=random_seed)
    print("random size cropped")

    # random flipping
    if flip:
      def _postprocess_flip(im):
        im = tf.image.random_flip_left_right(im, seed=random_seed)
        return im
      crop_image = _postprocess_flip(crop_image)
      print("flipped image")
      
  elif not is_training: # mode == EVAL

    if not shape_undefined:
      # central crop
      off = np.zeros(shape=[4])
      off[0] = int((now_size_x - crop_size) / 2)
      off[1] = int((now_size_y - crop_size) / 2)
      off[2:4] = off[:2] + crop_size
      off[0] = off[0] * 1.0/(now_size_x - 1)
      off[2] = off[2] * 1.0/(now_size_x - 1)
      off[1] = off[1] * 1.0/(now_size_y - 1)
      off[3] = off[3] * 1.0/(now_size_y - 1)
      off = np.reshape(off, shape=[1,4])

      box_ind = tf.constant(value=0, shape=[1], dtype=tf.in32)
      norm_im = tf.expand_dims(norm_im, axis=0)
      
      crop_image = tf.image.crop_and_resize(norm_im, boxes=off, box_ind=box_ind, crop_size=tf.constant([crop_size, crop_size]))
      crop_image = tf.squeeze(crop_image)
      
    else: # shape is undefined
      image = _aspect_preserving_resize(norm_im, 256)
      image = _central_crop([image], crop_size, crop_size)[0]
      image.set_shape([crop_size, crop_size, 3])
      crop_image = image

  if color_norm:
    crop_image = tf.div(crop_image, tf.constant(255, dtype=tf.float32))
    crop_image = ColorNormalize(crop_image)
    print("color normalized")

  if IMAGE_RESIZE is not None:
    print("resizing to", IMAGE_RESIZE)
    crop_image = tf.image.resize_images(crop_image, [IMAGE_RESIZE, IMAGE_RESIZE], align_corners=True)

  return crop_image
  

def random_sized_crop(image,
                      out_height,
                      out_width,
                      random_seed=3,
                      num_channels=3,
                      fix_aspect_ratio=False,
                      size_minval=0.08):

    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)

    area = height*width
    rnd_area = tf.random_uniform(shape=[1], minval=size_minval, maxval=1.0, dtype=tf.float32, seed=random_seed)[0] * area

    if fix_aspect_ratio:
        asp_ratio = 1
    else:
        asp_ratio = tf.random_uniform(shape=[1], minval=3.0/4, maxval=4.0/3, dtype=tf.float32, seed=random_seed)[0]
    
    crop_height = tf.sqrt(rnd_area * asp_ratio)
    crop_width = tf.sqrt(rnd_area / asp_ratio)
    div_ratio = tf.maximum(crop_height/height, tf.constant(1.0, dtype=tf.float32))
    div_ratio = tf.maximum(crop_width/width, div_ratio)

    crop_height = tf.cast(crop_height/div_ratio, tf.int32)
    crop_width = tf.cast(crop_width/div_ratio, tf.int32)

    crop_image = tf.random_crop(image, [crop_height, crop_width, num_channels])
    image = tf.image.resize_images(crop_image, [out_height, out_width])

    image.set_shape([out_height, out_width, num_channels])
    print("random size cropped an image")
    return image

def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image

def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width


def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    cGrop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs

# def _crop(image, offset_height, offset_width, crop_height, crop_width):
#   """Crops the given image using the provided offsets and sizes.

#   Note that the method doesn't assume we know the input image size but it does
#   assume we know the input image rank.

#   Args:
#     image: an image of shape [height, width, channels].
#     offset_height: a scalar tensor indicating the height offset.
#     offset_width: a scalar tensor indicating the width offset.
#     crop_height: the height of the cropped image.
#     crop_width: the width of the cropped image.

#   Returns:
#     the cropped (and resized) image.

#   Raises:
#     InvalidArgumentError: if the rank is not 3 or if the image dimensions are
#       less than the crop size.
#   """
#   original_shape = tf.shape(image)

#   rank_assertion = tf.Assert(
#       tf.equal(tf.rank(image), 3),
#       ['Rank of image must be equal to 3.'])
#   cropped_shape = control_flow_ops.with_dependencies(
#       [rank_assertion],
#       tf.stack([crop_height, crop_width, original_shape[2]]))

#   size_assertion = tf.Assert(
#       tf.logical_and(
#           tf.greater_equal(original_shape[0], crop_height),
#           tf.greater_equal(original_shape[1], crop_width)),
#       ['Crop size greater than the image size.'])

#   offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

#   # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
#   # define the crop size.
#   image = control_flow_ops.with_dependencies(
#       [size_assertion],
#       tf.slice(image, offsets, cropped_shape))
#   return tf.reshape(image, cropped_shape)

def ColorNormalize(image):
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - imagenet_mean) / imagenet_std

    return image
'''
Relative Position Preprocessing
'''

patch_sz = (96, 96)
gap = 48
noise = 7

def rp_preprocess_image(image, is_training, num_grids=4, g_noise=0, std=False, sub_mean=True):
    if is_training:
        return rp_preprocess_for_train(image, num_grids, g_noise, std, sub_mean=sub_mean)
    else:
        return rp_preprocess_for_val(image, std, sub_mean=sub_mean)

def rp_preprocess_for_train(image, num_grids=4, g_noise=0, std=False, sub_mean=True):

    num_grids = num_grids

    crop_images_0 = []
    crop_images_1 = []
    crop_images_2 = []
    crop_images_3 = []

    # do the substract mean
    if sub_mean:
        offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
        image -= offset
    
    if std:
        print("Using standard deviation preprocessing")
        scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
        image /= scale

    ratio = tf.div(tf.cast(tf.shape(image)[0], tf.float32),tf.cast(tf.shape(image)[1], tf.float32))
    #ratio = tf.Print(ratio, [ratio], message='Image ratio')
    #image = tf.Print(image, [tf.shape(image)], message='Image ratio')
    #print("ratio:", ratio)

    min_size = tf.cast(patch_sz[0] * 2 + gap + noise, tf.float32)
    scale_min_size = tf.maximum(min_size,  ratio * min_size)
    scale_min_size = tf.cast(scale_min_size, tf.int32)
    scale_0 = tf.random_uniform([], minval=tf.maximum(scale_min_size, tf.cast(tf.sqrt(tf.cast(tf.multiply(150000.0, ratio), tf.float32)), tf.int32)), \
            maxval=tf.maximum(scale_min_size + 1, tf.cast(tf.sqrt(tf.cast(tf.multiply(450000.0, ratio), tf.float32)), tf.int32)), dtype=tf.int32)
    scale_1 = tf.div(tf.cast(scale_0, tf.float32), ratio)
    scale_1 = tf.cast(tf.ceil(scale_1), tf.int32)

    image = tf.image.resize_images(image, [scale_0, scale_1])
    print("After resize:")
    print(image)
    for grid_inx in range(0, num_grids):

        crop_image = tf.random_crop(image, [patch_sz[0] * 2 + gap + noise, patch_sz[1] * 2 + gap + noise, 3])
        #print("crop_image:", crop_image)
        #print("shape_2:", tf.shape(crop_image)[2])

        for pair, crop_images in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [crop_images_0, crop_images_1, crop_images_2, crop_images_3]):
            expand_x = patch_sz[0] + gap - 1
            expand_y = patch_sz[1] + gap - 1
            startx = tf.random_uniform([], minval=pair[0] * expand_x, maxval=pair[0] * expand_x + 7, dtype=tf.int32)
            #print("startx:", startx)
            #print("scale_1:", scale_1)
            starty = tf.random_uniform([], minval=pair[1] * expand_y, maxval=pair[1] * expand_y + 7, dtype=tf.int32)

            _crop_images = _crop(crop_image, startx, starty, patch_sz[0], patch_sz[1])
            _crop_images.set_shape(_crop_images.get_shape().as_list()[:-1] + [3])
            
            if g_noise==1:
                drop_channel_0 = tf.random_uniform([], minval=0, maxval=3, dtype=tf.int32)
                drop_channel_1 = (drop_channel_0 + 1) % 3
                remaining_channel = (drop_channel_0 + 2) % 3
                remaining_layer = tf.expand_dims(_crop_images[:,:,remaining_channel], axis=2)
                #print("old_images:", _crop_images)
                #print("remaining_layer:", remaining_layer)
                drop_layer_1 = tf.random_uniform(shape=tf.shape(_crop_images[:,:,drop_channel_0]), dtype=tf.float32) - 0.5 
                drop_layer_1 = tf.expand_dims(drop_layer_1, axis=2)
                drop_layer_2 = tf.random_uniform(shape=tf.shape(_crop_images[:,:,drop_channel_0]), dtype=tf.float32) - 0.5
                drop_layer_2 = tf.expand_dims(drop_layer_2, axis=2)

                _crop_images = tf.cond(pred=tf.equal(remaining_channel, 0), true_fn=lambda:tf.concat([remaining_layer, drop_layer_1, drop_layer_2], 2), false_fn=lambda:_crop_images)
                _crop_images = tf.cond(pred=tf.equal(remaining_channel, 1), true_fn=lambda:tf.concat([drop_layer_2, remaining_layer, drop_layer_1], 2), false_fn=lambda:_crop_images)
                _crop_images = tf.cond(pred=tf.equal(remaining_channel, 2), true_fn=lambda:tf.concat([drop_layer_1, drop_layer_2, remaining_layer], 2), false_fn=lambda:_crop_images)

            elif g_noise==2:
                drop_channel_0 = tf.random_uniform([], minval=0, maxval=3, dtype=tf.int32)
                drop_channel_1 = (drop_channel_0 + 1) % 3
                remaining_channel = (drop_channel_0 + 2) % 3
                remaining_layer = tf.expand_dims(_crop_images[:,:,remaining_channel], axis=2)
                #print("old_images:", _crop_images)
                #print("remaining_layer:", remaining_layer)
                drop_layer_1 = tf.random_normal(shape=tf.shape(_crop_images[:,:,drop_channel_0]), dtype=tf.float32)
                drop_layer_1 = tf.expand_dims(drop_layer_1, axis=2)
                drop_layer_2 = tf.random_normal(shape=tf.shape(_crop_images[:,:,drop_channel_0]), dtype=tf.float32)
                drop_layer_2 = tf.expand_dims(drop_layer_2, axis=2)

                _crop_images = tf.cond(pred=tf.equal(remaining_channel, 0), true_fn=lambda:tf.concat([remaining_layer, drop_layer_1, drop_layer_2], 2), false_fn=lambda:_crop_images)
                _crop_images = tf.cond(pred=tf.equal(remaining_channel, 1), true_fn=lambda:tf.concat([drop_layer_2, remaining_layer, drop_layer_1], 2), false_fn=lambda:_crop_images)
                _crop_images = tf.cond(pred=tf.equal(remaining_channel, 2), true_fn=lambda:tf.concat([drop_layer_1, drop_layer_2, remaining_layer], 2), false_fn=lambda:_crop_images)


                #print("new_images:", _crop_images)
            crop_images.append(_crop_images)
            #print("crop_patch:", _crop_images)


    # print("output image shape:", new_images_0.shape, new_images_1.shape)
    new_images = tf.stack(
            [crop_images_0, crop_images_1, crop_images_2, crop_images_3], 
            axis=1)
    print("*****new_images shape******")
    print(new_images.shape)

    return new_images

def rp_preprocess_for_val(image, std=False, sub_mean=True):

    crop_images_0 = []
    crop_images_1 = []
    crop_images_2 = []
    crop_images_3 = []
   
    # do the substract mean
    if sub_mean:
        offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
        image -= offset

    if std:
        print("Using standard deviation preprocessing")
        scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
        image /= scale

    g_noise = 0 
    image = _do_scale(image, COL_IMAGE_SIZE + 32)
 
    for grid_inx in range(0, 1):
        crop_image = _center_crop(image, patch_sz[0] * 2 + gap + noise)
        #crop_image = tf.random_crop(image, [patch_sz[0] * 2 + gap + noise, patch_sz[1] * 2 + gap + noise, 3])
        #print("crop_image:", crop_image)
        #print("shape_2:", tf.shape(crop_image)[2])

        for pair, crop_images in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [crop_images_0, crop_images_1, crop_images_2, crop_images_3]):
            expand_x = patch_sz[0] + gap - 1
            expand_y = patch_sz[1] + gap - 1
            startx = pair[0] * expand_x
            starty = pair[1] * expand_y
            _crop_images = _crop(crop_image, startx, starty, patch_sz[0], patch_sz[1])
            _crop_images.set_shape(_crop_images.get_shape().as_list()[:-1] + [3])
            crop_images.append(_crop_images)


    new_images = tf.stack(
            [crop_images_0, crop_images_1, crop_images_2, crop_images_3],
            axis=1)
    print("*****new_images shape******")
    print(new_images.shape)

    return new_images

def col_preprocess_image(image, is_training=False, soft=True, down_sample=8, col_knn=False, col_tl=False, combine_rp=False):
  """Preprocesses the given image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.

  Returns:
    A preprocessed image `Tensor`.
  """
  if is_training:
    return col_preprocess_for_train(image, soft=soft, down_sample=down_sample, col_knn=col_knn, col_tl=col_tl, combine_rp=combine_rp)
  else:
    return col_preprocess_for_eval(image, soft=soft, down_sample=down_sample, col_knn=col_knn, col_tl=col_tl, combine_rp=combine_rp)


def col_preprocess_for_train(image, soft=True, down_sample=8, col_knn=False, col_tl=False, combine_rp=False):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.

  Returns:
    A preprocessed image `Tensor`.
  """
  if col_tl:
      COL_IMAGE_SIZE = 224
  else:
      COL_IMAGE_SIZE = 256
  # image = _random_crop(image, IMAGE_SIZE)
  image = random_sized_crop(image, COL_IMAGE_SIZE, COL_IMAGE_SIZE)
  #image = _normalize(image)
  image = _flip(image)
  image = tf.reshape(image, [COL_IMAGE_SIZE, COL_IMAGE_SIZE, 3])
    
  lab_image = rgb_to_lab(image)
  #lab_image = color.rgb2lab(image)
  
  l_image = lab_image[ :, :, :1]
  ab_image = lab_image[ :, :, 1:]

  l_image = l_image - 50 

  ab_image_ss = ab_image[::down_sample, ::down_sample, :]

  #Q_label = _nnencode(ab_image_ss)

  Q_label = ab_to_Q(ab_image_ss, soft=soft, col_knn=col_knn)
  
  #Q_label = Q_label_[0:-1:down_sample, 0:-1:down_sample, :] 
  if combine_rp:
      l_image = tf.tile(l_image, [1, 1, 3])
  return l_image, Q_label

def col_preprocess_for_eval(image, soft=True, down_sample=8, col_knn=False, col_tl=False, combine_rp=False):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.

  Returns:
    A preprocessed image `Tensor`.
  """
  if col_tl:
      COL_IMAGE_SIZE = 224
  else:
      COL_IMAGE_SIZE = 256
  image = _do_scale(image, COL_IMAGE_SIZE + 32)
  #image = _normalize(image)
  image = _center_crop(image, COL_IMAGE_SIZE)
  image = tf.reshape(image, [COL_IMAGE_SIZE, COL_IMAGE_SIZE, 3])
  
  lab_image = rgb_to_lab(image)
  l_image = lab_image[ :, :, :1]

  l_image = l_image - 50

  ab_image = lab_image[ :, :, 1:]
  ab_image_ss = ab_image[::down_sample, ::down_sample, :]

  Q_label = ab_to_Q(ab_image_ss, soft=soft, col_knn=col_knn)

  #Q_label = Q_label_[0:-1:down_sample, 0:-1:down_sample, :] 
  if combine_rp:
      l_image = tf.tile(l_image, [1, 1, 3])

  return l_image, Q_label


def col_preprocess_for_gpu(image, soft=True, down_sample=4, col_knn=False):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.

  Returns:
    A preprocessed image `Tensor`.
  """
  # image = _random_crop(image, IMAGE_SIZE)
  # image = random_sized_crop(image, COL_IMAGE_SIZE, COL_IMAGE_SIZE)
  #image = _normalize(image)
  #image = _flip(image)
  #image = tf.reshape(image, [COL_IMAGE_SIZE, COL_IMAGE_SIZE, 3])
  #image = tf.div(image, tf.constant(255, dtype=tf.float32))
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  #image = tf.Print(image, [image], message='original image')
  lab_image = rgb_to_lab(image)
  #lab_image = color.rgb2lab(image)
  
  l_image = lab_image[:, :, :, :1]
  ab_image = lab_image[:, :, :, 1:]

  l_image = l_image - 50
  #l_image = tf.Print(l_image, [l_image], message='l_image')
  ab_image_ss = ab_image[:, ::down_sample, ::down_sample, :]
  Q_label = ab_to_Q(ab_image_ss, soft=soft, col_knn=col_knn)
  
  #Q_label_ = ab_to_Q(ab_image, soft=soft)
  #Q_label = Q_label_[:, ::down_sample, ::down_sample, :] 
 
  return l_image, ab_image, Q_label


def depth_image_preprocess_for_train(original_image, depth_image, image_height, image_width, down_sample=1, color_dp_tl=False, combine_3_task=False):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.

  Returns:
G    A preprocessed image `Tensor`.
  """
  # image = _random_crop(image, IMAGE_SIZE)
  random_height = tf.random_uniform([], minval=0, maxval=image_height-224, dtype=tf.int32)
  random_width = tf.random_uniform([], minval=0, maxval=image_width-224, dtype=tf.int32)

  crop_original_image = tf.slice(original_image, [random_height, random_width, 0], [224, 224, 3])
  crop_depth_image = tf.slice(depth_image, [random_height, random_width, 0], [224, 224, 1])

  #original_image = tf.random_crop(original_image, [IMAGE_SIZE, IMAGE_SIZE, 3], seed=3)
  #depth_image = tf.random_crop(depth_image, [IMAGE_SIZE, IMAGE_SIZE, 1], seed=3)
  
  random_flip = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)
  
  def image_flip(o_image, d_image):
    o_image_f = tf.image.flip_left_right(o_image)
    d_image_f = tf.image.flip_left_right(d_image)
    return o_image_f, d_image_f
  
  def image_identity(o_image, d_image):
      return o_image, d_image
  
  flip_original_image, flip_depth_image = tf.cond(random_flip < 1, lambda: image_flip(crop_original_image, crop_depth_image), 
          lambda: image_identity(crop_original_image, crop_depth_image))

  #original_image = _flip(original_image)
  #depth_image = _flip(depth_image)  
  #orginal_image = _center_crop(original_image, IMAGE_SIZE)
  #depth_image = _center_crop(depth_image, IMAGE_SIZE)

  original_image = tf.reshape(flip_original_image, [224, 224, 3])
  depth_image = tf.reshape(flip_depth_image, [224, 224, 1])

  if color_dp_tl:
      lab_image = rgb_to_lab(original_image)
      original_image = lab_image[:, :, :1] - 50
      if combine_3_task:
          original_image = tf.tile(original_image, [1, 1, 3])

  if down_sample > 1:
      depth_image = depth_image[::down_sample, ::down_sample, :]
  
  return original_image, depth_image

def depth_image_preprocess_for_val(original_image, depth_image, image_height, image_width, down_sample=1, color_dp_tl=False, combine_3_task=False):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.

  Returns:
    A preprocessed image `Tensor`.
  """
  #original_image = _do_scale(original_image, IMAGE_SIZE + 32)
  #depth_image = _do_scale(depth_image, IMAGE_SIZE + 32)
   
  start_height = int((image_height - 224) / 2)
  start_width = int((image_width - 224) / 2)

  original_image = tf.slice(original_image, [start_height, start_width, 0], [224, 224, 3])
  depth_image = tf.slice(depth_image, [start_height, start_width, 0], [224, 224, 1])

  #original_image = _center_crop(original_image, IMAGE_SIZE)
  #depth_image = _center_crop(depth_image, IMAGE_SIZE)
  
  original_image = tf.reshape(original_image, [224, 224, 3])
  depth_image = tf.reshape(depth_image, [224, 224, 1])
  if color_dp_tl:
      lab_image = rgb_to_lab(original_image)
      original_image = lab_image[:, :, :1] - 50
      if combine_3_task:
          original_image = tf.tile(original_image, [1, 1, 3])

  if down_sample > 1:
      depth_image = depth_image[::down_sample, ::down_sample, :]

  return original_image, depth_image

def depth_image_preprocess_image(original_image, depth_image, is_training, image_height, image_width, down_sample=1, color_dp_tl=False, combine_3_task=False):
  """Preprocesses the given image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.

  Returns:
    A preprocessed image `Tensor`.
  """
  if is_training:
      original_image, depth_image = depth_image_preprocess_for_train(original_image, depth_image, image_height, image_width, down_sample=down_sample, color_dp_tl=color_dp_tl, combine_3_task=combine_3_task)
  else:
      original_image, depth_image = depth_image_preprocess_for_val(original_image, depth_image, image_height, image_width, down_sample=down_sample, color_dp_tl=color_dp_tl, combine_3_task=combine_3_task)

  return original_image, depth_image

def depth_preprocess_image(image, ab_depth=False):
  """Preprocesses the given image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.

  Returns:
    A preprocessed image `Tensor`.
  """
  if ab_depth:
    image = tf.cast(image, dtype=tf.float32)
    image = tf.div(image, tf.constant(8000, dtype=tf.float32))  
  else:
    image = tf.image.per_image_standardization(image)

  return image
  
    

def rp_col_preprocess_image(image, is_training=False, down_sample=8, col_knn=True):

  if is_training:
    return rp_col_preprocess_for_train(image, down_sample=down_sample, col_knn=col_knn)
  else:
    return rp_col_preprocess_for_eval(image, down_sample=down_sample, col_knn=col_knn)

def rp_col_preprocess_for_train(image, down_sample=8, col_knn=True):
      
  lab_image = rgb_to_lab(image)
  #lab_image = color.rgb2lab(image)
  
  l_image = lab_image[ :, :, :1]

  l_image = l_image - 50 
  l_image = tf.tile(l_image, [1, 1, 3])
  return l_image

def rp_col_preprocess_for_eval(image, down_sample=8, col_knn=True):
    
  lab_image = rgb_to_lab(image)
  l_image = lab_image[ :, :, :1]

  l_image = l_image - 50
  l_image = tf.tile(l_image, [1, 1, 3])

  return l_image

