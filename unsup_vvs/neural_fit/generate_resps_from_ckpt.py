from unsup_vvs.neural_fit.cleaned_network_builder import get_network_outputs
from unsup_vvs.neural_fit.brainscore_mask.bs_fit_utils import get_load_settings_from_func
import argparse
from argparse import Namespace
import os
import importlib
import tensorflow as tf
import json
import sys
import numpy as np
import pdb
sys.path.append('../network_training/')


def load_set_func(args):
    all_paths = args.set_func.split('.')
    module_name = '.'.join(
            ['unsup_vvs', 'neural_fit', 'brainscore_mask', 'settings'] \
            + all_paths[:-1])
    load_setting_module = importlib.import_module(module_name)
    set_func = all_paths[-1]
    set_func = getattr(load_setting_module, set_func)
    args = set_func(args)
    return args


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to generate responses from ckpts')
    parser.add_argument(
            '--gpu', default='0', type=str, action='store')
    parser.add_argument(
            '--set_func', type=str, 
            default=None, required=True,
            action='store')
    parser.add_argument(
            '--ckpt_path', 
            default=None, required=True,
            type=str, action='store',
            help='Ckpt path to load from')
    parser.add_argument(
            '--model_type', default='vm_model', type=str, action='store')
    parser.add_argument(
            '--prep_type', default='mean_std', type=str, action='store')
    parser.add_argument(
            '--setting_name', default=None, type=str, action='store',
            help='Network setting name')
    parser.add_argument(
            '--cfg_kwargs', default="{}", type=str, action='store',
            help='Kwargs for network cfg')
    return parser


def test_get_response():
    parser = get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args = load_set_func(args)

    batch_size = 1
    img_placeholder = tf.placeholder(
            dtype=tf.uint8, 
            shape=[batch_size, 224, 224, 3])
    network_outputs, _ = get_network_outputs(
                {'images': img_placeholder},
                prep_type=args.prep_type,
                model_type=args.model_type,
                setting_name=args.setting_name,
                **json.loads(args.cfg_kwargs))

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    SESS = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options,
            ))

    # This should be the actual input images
    input_images = np.zeros([batch_size, 224, 224, 3], dtype=np.uint8)
    saver.restore(SESS, args.ckpt_path)
    outputs_np = SESS.run(
            network_outputs, 
            feed_dict={img_placeholder: input_images})
    pdb.set_trace()
    pass


if __name__ == '__main__':
    test_get_response()
