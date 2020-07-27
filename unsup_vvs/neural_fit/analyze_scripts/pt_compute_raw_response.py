import argparse
import copy
import torch
import pdb
import torch.backends.cudnn as cudnn
import os
import sys
import json
import numpy as np
import pickle
from torch import nn
from tqdm import tqdm
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./brainscore_mask'))
from cleaned_network_builder import get_network_outputs
from brainscore_mask import tf_model_loader
import bs_fit_neural as bs_fit
import circular_var
import stimuli_compute
import pt_stimuli_compute
import raw_stimuli_compute


def get_pt_compute_raw_response_parser():
    parser = argparse.ArgumentParser(
            description='Compute the raw response for pytorch models')
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
    parser.add_argument(
            '--start_idx', default=0, type=int, action='store',
            help='Batch size')
    parser = bs_fit.add_load_settings(parser)
    parser = bs_fit.add_model_settings(parser)
    return parser


class PtComputeRawResponse(pt_stimuli_compute.PtStimuliCompute):
    def __init__(self, args):
        self.args = args
        self.start_idx = args.start_idx
        self.batch_size = args.batch_size
        self.device = torch.device("cuda")
        self.build_model()
        pass

    def build_input_images(self):
        pass

    def get_normalized_image(self):
        image = self.images.permute(0, 3, 1, 2)
        image = image / 255
        image = torch.clamp(image, min=0.0, max=1.0)
        image = (image - self._mean) / self._std
        return image

    def get_predictions(self, curr_output):
        predictions = curr_output.mean(dim=(2, 3))
        predictions = predictions[:, self.start_idx : self.start_idx + self.batch_size]
        return predictions

    def get_output(self, images):
        layer = self.args.layer
        self.layer = layer
        result_list = []
        _layer_module = self.get_layer(layer)
        _layer_module.register_forward_hook(
                lambda _layer, _input, output: result_list.append(output))

        self.images = torch.from_numpy(images).to(self.device, torch.float)
        image = self.get_normalized_image()
        image = self.add_preprocess(image)
        self._model(image)
        curr_output = result_list.pop()
        predictions = self.get_predictions(curr_output)
        return predictions.cpu().data.numpy()


def main():
    parser = get_pt_compute_raw_response_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.set_func, "Must specify set_func"
    args = bs_fit.load_set_func(args)

    pt_compute_raw_response = PtComputeRawResponse(args)
    input_images = np.load(args.file_path)[:args.batch_size]
    output = pt_compute_raw_response.get_output(input_images)
    print(np.trace(output[:, :args.batch_size]))


if __name__ == '__main__':
    main()
