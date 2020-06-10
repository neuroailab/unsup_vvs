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
import bs_fit_neural as bs_fit
import bs_fit_utils
import circular_var
import stimuli_compute


def get_pt_stimuli_compute_parser():
    parser = argparse.ArgumentParser(
            description='The script to compute the optimal stimuli for pytorch models')
    parser = stimuli_compute.add_stimuli_settings(parser)
    parser = bs_fit.add_load_settings(parser)
    parser = bs_fit.add_model_settings(parser)
    return parser


class PtStimuliCompute(object):
    def __init__(self, args, start_idx=0, batch_size=None):
        self.args = args
        self.which_split = 0
        self.id_suffix = '-' + args.id_suffix
        self.bench = stimuli_compute.REGION_TO_BENCH_MAP[args.region]
        self.start_idx = start_idx
        self.batch_size = batch_size
        self.device = torch.device("cuda")
        self.build_model()
        pass

    def __initialize_images(self):
        nn.init.uniform_(self.images_var, -0.5, 0.5)

    def build_input_images(self):
        args = self.args
        input_resolution = stimuli_compute.REGION_TO_INPUT_RESOLUTION_MAP[args.region]
        num_of_images = stimuli_compute.REGION_TO_OUTSHAPE_MAP[args.region]
        if args.just_raw:
            num_of_images = 64
        if self.batch_size is not None:
            num_of_images = self.batch_size
        image_shape = (
                num_of_images,
                input_resolution, input_resolution, 
                1)
        self.images_var = torch.empty(*image_shape).to(
                self.device, torch.float).requires_grad_()

    def __build_dc_model(self):
        args = self.args
        assert getattr(args, 'load_from_ckpt', None) is not None, \
                "Must specify ckpt to load from"
        dc_model = bs_fit.get_dc_model(
                args.load_from_ckpt, verbose=False)
        self._model = dc_model.features
        self.add_preprocess = dc_model.sobel
        self.layers = bs_fit.PT_RES18_LAYERS
        self.model_id = args.identifier

    def __build_la_cmc_model(self):
        from pt_scripts.main import tolab_normalize, LAB_MEAN, LAB_STD
        args = self.args
        assert getattr(args, 'load_from_ckpt', None) is not None, \
                "Must specify ckpt to load from"
        la_cmc_model = bs_fit_utils.get_la_cmc_model(args.load_from_ckpt)
        self._model = la_cmc_model.module.l_to_ab
        self.layers = bs_fit.PT_RES18_LAYERS
        self.model_id = args.identifier
        def _preprocess(img):
            denorm_img = img * self._std + self._mean
            lab_img = bs_fit_utils.xyz2lab(bs_fit_utils.rgb2xyz(denorm_img))
            lab_mean = torch.tensor(LAB_MEAN).to(self.device).view(-1, 1, 1)
            lab_std = torch.tensor(LAB_STD).to(self.device).view(-1, 1, 1)
            lab_img = (lab_img - lab_mean) / lab_std
            lab_img = lab_img[:, :1, :, :]
            return lab_img
        self.add_preprocess = _preprocess

    def build_model(self):
        self._mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(-1, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(-1, 1, 1)
        self.build_input_images()

        pt_model_type = getattr(self.args, 'pt_model', None)
        if pt_model_type == 'deepcluster':
            self.__build_dc_model()
        elif pt_model_type == 'la_cmc':
            self.__build_la_cmc_model()
        else:
            raise NotImplementedError

    def get_layer(self, layer_name):
        module = self._model
        for part in layer_name.split('.'):
            module = module._modules.get(part)
            assert module is not None, \
                    f"No submodule found for layer {layer_name}, at part {part}"
        return module

    def get_normalized_image(self):
        image = self.images_var.permute(0, 3, 1, 2)
        image = image + 0.5
        image = torch.clamp(image, min=0.0, max=1.0)
        image = image.repeat(1, 3, 1, 1)
        image = (image - self._mean) / self._std
        return image

    def get_uint_scale_image(self):
        image = (self.images_var + 0.5) * 255
        image = torch.clamp(image, min=0.0, max=255.0)
        return image

    def _build_mask_predictor(self):
        mask_ckpt_path = circular_var.FIT_CKPT_PATH_PATTERN.format(
                model_id = self.model_id,
                id_suffix = self.id_suffix,
                layer = self.layer,
                bench = self.bench,
                which_split = self.which_split,
                )
        from tensorflow.python import pywrap_tensorflow
        reader = pywrap_tensorflow.NewCheckpointReader(mask_ckpt_path)
        s_mask = reader.get_tensor('mapping/spatial_mask')
        d_mask = reader.get_tensor('mapping/depth_mask')
        bias = reader.get_tensor('mapping/bias')

        s_mask = torch.from_numpy(s_mask).to(self.device, torch.float)
        s_mask.requires_grad = False
        s_mask = s_mask.permute(2, 0, 1, 3)

        d_mask = torch.from_numpy(d_mask).to(self.device, torch.float)
        d_mask.requires_grad = False
        d_mask = d_mask.permute(2, 0, 1, 3)

        bias = torch.from_numpy(bias).to(self.device, torch.float)

        self.mask_weight = s_mask * d_mask
        self.mask_weight = self.mask_weight.view(-1, self.mask_weight.shape[-1])
        self.mask_bias = bias

    def get_predictions(self, curr_output):
        curr_output = curr_output.view(curr_output.shape[0], -1)
        predictions = torch.matmul(curr_output, self.mask_weight)
        predictions = predictions + self.mask_bias
        if self.batch_size is not None:
            predictions = predictions[:, self.start_idx : self.start_idx + self.batch_size]
        return predictions

    def get_loss(self, predictions):
        args = self.args
        loss = -predictions.trace()
        total_variance = \
                torch.sum(torch.abs(self.images_var[:, :-1, :, :] - self.images_var[:, 1:, :, :])) \
                + torch.sum(torch.abs(self.images_var[:, :, :-1, :] - self.images_var[:, :, 1:, :]))
        loss += total_variance * args.tv_wd
        return loss

    def get_optimizer(self):
        args = self.args
        optimizer = torch.optim.Adam(
                [self.images_var], lr=args.lr, 
                weight_decay=args.wd)
        return optimizer

    def compute_optimal_stimuli(self, layer):
        self.layer = layer
        self.__initialize_images()
        optimizer = self.get_optimizer()

        result_list = []
        _layer_module = self.get_layer(layer)
        _layer_module.register_forward_hook(
                lambda _layer, _input, output: result_list.append(output))

        self._build_mask_predictor()

        loss_vals = []
        for _ in range(stimuli_compute.NUM_STEPS):
            image = self.get_normalized_image()
            image = self.add_preprocess(image)
            self._model(image)

            curr_output = result_list.pop()
            predictions = self.get_predictions(curr_output)
            loss = self.get_loss(predictions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_vals.append(loss.item())

        save_result = {
                'loss_vals': loss_vals,
                'images': self.get_uint_scale_image().cpu().data.numpy(),
                'actual_response': predictions.cpu().data.numpy().diagonal(),
                }
        del optimizer
        self.clean_ops()
        return save_result

    def clean_ops(self):
        del self.mask_weight
        del self.mask_bias

    def dump_result(self, save_result):
        special = self.args.special or '_tv_jitter'
        result_path = stimuli_compute.RESULT_PATH_PATTERN.format(
                model_id = self.model_id,
                id_suffix = self.id_suffix,
                layer = self.layer,
                bench = self.bench,
                which_split = self.which_split,
                special = special,
                )
        save_dir = os.path.dirname(result_path)
        if not os.path.isdir(save_dir):
            os.system('mkdir -p ' + save_dir)
        pickle.dump(save_result, open(result_path, 'wb'))

    def close(self):
        pass
    

def compute_aggregate_dump(args, builder):
    list_of_all_results = []
    num_neurons = stimuli_compute.REGION_TO_OUTSHAPE_MAP[args.region]
    all_stimuli_compute = builder(args)
    for start_idx in range(0, num_neurons, args.batch_size):
        all_results = {}
        batch_size = min(args.batch_size, num_neurons - start_idx)
        layers = all_stimuli_compute.layers
        if args.just_raw:
            layers = layers[:2]
        if args.layers is not None:
            layers = args.layers.split(',')

        for which_split in range(stimuli_compute.NUM_SPLITS):
            if which_split not in all_results:
                all_results[which_split] = {}

            for layer in tqdm(layers):
                each_stimuli_compute = builder(
                        args, start_idx=start_idx, 
                        batch_size=batch_size)
                each_stimuli_compute.which_split = which_split
                all_results[which_split][layer] = each_stimuli_compute.compute_optimal_stimuli(layer)
                del each_stimuli_compute

        list_of_all_results.append(all_results)

    final_result = stimuli_compute.aggregate_results(list_of_all_results)
    for which_split in range(stimuli_compute.NUM_SPLITS):
        for layer in final_result[which_split]:
            all_stimuli_compute.layer = layer
            all_stimuli_compute.which_split = which_split
            all_stimuli_compute.dump_result(final_result[which_split][layer])


def main():
    parser = get_pt_stimuli_compute_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.set_func, "Must specify set_func"
    args = bs_fit.load_set_func(args)

    compute_aggregate_dump(args, builder=PtStimuliCompute)


if __name__ == '__main__':
    main()
