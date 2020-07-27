import argparse
import copy
import tensorflow as tf
import pdb
from tqdm import tqdm
import os
import sys
import json
import numpy as np
import pickle
sys.path.append('../combine_pred/')
import cmd_parser
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./brainscore_mask'))
from cleaned_network_builder import get_network_outputs
from brainscore_mask import tf_model_loader
import bs_fit_neural as bs_fit
import circular_var
import stimuli_compute


def get_compute_raw_response_parser():
    parser = argparse.ArgumentParser(
            description='Compute the response for inputs')
    parser.add_argument(
            '--gpu', default='0', type=str, action='store')
    parser.add_argument(
            '--file_path', type=str, action='store', 
            required=True)
    parser.add_argument(
            '--layer', type=str, action='store', 
            required=True)
    parser.add_argument(
            '--batch_size', default=64, type=int, action='store',
            help='Batch size')
    parser = bs_fit.add_load_settings(parser)
    parser = bs_fit.add_model_settings(parser)
    return parser


class RawStimuliCompute(stimuli_compute.StimuliCompute):
    def __init__(self, args):
        self.args = args
        self.build_model()
        self.build_output()

    def build_input_images(self):
        args = self.args
        batch_size = args.batch_size
        input_shape = 224
        self.images = tf.placeholder(
                tf.uint8, 
                shape=(batch_size, input_shape, input_shape, 3),
                name='inputs')

    def build_output(self):
        layer = self.args.layer
        assert layer in self.ending_points
        _raw_output = self.ending_points[layer]
        self.output = tf.reduce_mean(_raw_output, axis=(1, 2))

    def get_output(self, images):
        output = self.SESS.run(self.output, feed_dict={self.images: images})
        return output


def main():
    parser = get_compute_raw_response_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.set_func, "Must specify set_func"
    args = bs_fit.load_set_func(args)

    raw_stimuli_compute = RawStimuliCompute(args)
    input_images = np.load(args.file_path)[:args.batch_size]
    output = raw_stimuli_compute.get_output(input_images)
    print(np.trace(output[:, :args.batch_size]))


if __name__ == '__main__':
    main()
