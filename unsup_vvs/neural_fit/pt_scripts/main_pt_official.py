import argparse
import pdb
import torch
import torch.backends.cudnn as cudnn
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import torch.nn as nn
try:
    import cPickle
    pickle = cPickle
except:
    import pickle
import sys
import pt_official_model
import main


def get_parser():
    parser = argparse.ArgumentParser(
            description='Generate outputs for neural fitting')

    parser.add_argument(
            '--data', type=str, 
            default='/mnt/fs0/datasets/neural_data'\
                    + '/img_split/V4IT/tf_records/images',
            help='path to stimuli')
    parser.add_argument(
            '--batch_size', default=32, type=int,
            help='mini-batch size')
    parser.add_argument(
            '--save_path', type=str,
            default='/mnt/fs4/chengxuz/v4it_temp_results/'\
                    + 'pt_official_res18_nf/V4IT_split_0',
            help='path for storing results')
    parser.add_argument(
            '--dataset_type', type=str, default='hvm')
    return parser


def color_normalize(image):
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - imagenet_mean) / imagenet_std
    return image


class NfPtOfficialOutput(main.NfOutput):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu")
        self.need_to_make_meta = True
        self.model = pt_official_model.PtOfficialModel()
        self.all_writers = None

    def get_one_image(self, string_record):
        example = tf.train.Example()
        example.ParseFromString(string_record)
        img_string = (example.features.feature['images']
                                      .bytes_list
                                      .value[0])
        img_array = np.fromstring(img_string, dtype=np.float32)
        img_array = img_array.reshape(main.ORIGINAL_SHAPE)
        img_array *= 255
        img_array = img_array.astype(np.uint8)
        img_array = np.array(
                Image.fromarray(img_array).resize(main.INPUT_SHAPE[:2]))
        img_array = img_array.astype(np.float32)
        img_array /= 255.
        img_array = color_normalize(img_array)
        img_array = np.transpose(img_array, [2, 0, 1])
        img_array = img_array.astype(np.float32)
        return img_array


def main_func():
    parser = get_parser()
    args = parser.parse_args()

    if args.dataset_type == 'v1_tc':
        main.TFR_PAT = 'split'
    if args.dataset_type == 'hvm_var6':
        main.TFR_PAT = 'of'

    all_tfr_path = main.get_tfr_files(args.data)

    nf_output = NfPtOfficialOutput(args)
    
    for tfr_path in tqdm(all_tfr_path):
        nf_output.write_outputs_for_one_tfr(tfr_path)


if __name__ == '__main__':
    main_func()
