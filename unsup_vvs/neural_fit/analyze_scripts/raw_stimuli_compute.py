import argparse
import copy
import tensorflow as tf
import torch
import torch.backends.cudnn as cudnn
import pdb
from tqdm import tqdm
import os
import sys
import json
import numpy as np
import pickle
from model_tools.activations.pytorch import PytorchWrapper
sys.path.append('../combine_pred/')
import cmd_parser
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./brainscore_mask'))
from cleaned_network_builder import get_network_outputs
from brainscore_mask import tf_model_loader
import bs_fit_neural as bs_fit
import circular_var
import stimuli_compute
import xforms
RESULT_PATH_PATTERN = os.path.join(
        '/mnt/fs4/chengxuz/v4it_temp_results/.result_caching',
        'optimal_stimuli',
        'model_id={model_id}_raw',
        '{layer}{special}.pkl',
        )


def add_raw_stimuli_settings(parser):
    parser.add_argument(
            '--gpu', default='0', type=str, action='store')
    parser.add_argument(
            '--lr', default=0.05, type=float, action='store')
    parser.add_argument(
            '--wd', default=1e-4, type=float, action='store')
    parser.add_argument(
            '--tv_wd', default=1e-4, type=float, action='store')
    parser.add_argument(
            '--batch_size', default=32, type=int, action='store',
            help='Batch size')
    parser.add_argument(
            '--layers', type=str, 
            default=None,
            action='store')
    parser.add_argument(
            '--special', default=None, type=str, action='store')
    return parser


def get_raw_stimuli_compute_parser():
    parser = argparse.ArgumentParser(
            description='The script to compute the optimal stimuli for raw units')
    parser = add_raw_stimuli_settings(parser)
    parser = bs_fit.add_load_settings(parser)
    parser = bs_fit.add_model_settings(parser)
    return parser


class RawStimuliCompute(stimuli_compute.StimuliCompute):
    def __init__(self, args, start_idx=0, batch_size=None):
        self.args = args
        self.start_idx = start_idx
        self.batch_size = batch_size
        self.build_model()
        self.which_split = 0

    def build_input_images(self):
        args = self.args
        image_initializer = tf.random_uniform_initializer(-0.5, 0.5)
        image_regularizer = tf.contrib.layers.l2_regularizer(args.wd)

        input_resolution = 224
        num_of_images = self.batch_size
        image_shape = (
                num_of_images,
                input_resolution, input_resolution, 
                3)
        images = tf.get_variable(
                "images",
                image_shape,
                initializer=image_initializer,
                regularizer=image_regularizer)
        self.images_var = images

        scales = [1 + (i - 5) / 50. for i in range(11)]
        angles = list(range(-10, 11)) + 5 * [0]
        images = xforms.pad(images, pad_amount=12)
        images = xforms.jitter(images, jitter_amount=8)
        images = xforms.random_scale(images, scales)
        images = xforms.random_rotate(images, angles)
        images = xforms.jitter(images, jitter_amount=4)

        images = (images + 0.5) * 255
        images = tf.clip_by_value(images, 0, 255)
        self.images = images

    def build_tf_predictions(self):
        layer = self.layer
        assert layer in self.ending_points
        _raw_predictions = self.ending_points[layer]
        self._predictions = tf.reduce_mean(_raw_predictions, axis=(1, 2))
        self._predictions = self._predictions[:, self.start_idx : self.start_idx + self.batch_size]

    def build_loss(self):
        self.build_tf_predictions()
        self.set_vars()
        loss = tf.linalg.trace(self._predictions)
        self.loss = tf.negative(loss) + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        total_variation = self.get_total_var_loss()
        self.loss += self.args.tv_wd * total_variation

    def dump_result(self, save_result):
        special = self.args.special or '' 
        result_path = RESULT_PATH_PATTERN.format(
                model_id = self.model_id,
                layer = self.layer,
                special = special,
                )
        save_dir = os.path.dirname(result_path)
        if not os.path.isdir(save_dir):
            os.system('mkdir -p ' + save_dir)
        pickle.dump(save_result, open(result_path, 'wb'))


def main():
    parser = get_raw_stimuli_compute_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.set_func, "Must specify set_func"
    args = bs_fit.load_set_func(args)

    raw_stimuli_compute = RawStimuliCompute(
            args, start_idx=0, 
            batch_size=args.batch_size)
    layers = raw_stimuli_compute.layers
    for layer in tqdm(layers):
        _temp_result = raw_stimuli_compute.compute_optimal_stimuli(layer)
        raw_stimuli_compute.dump_result(_temp_result)
    raw_stimuli_compute.close()


if __name__ == '__main__':
    main()
