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


def get_raw_pt_stimuli_compute_parser():
    parser = argparse.ArgumentParser(
            description='Compute the optimal stimuli for pytorch models')
    parser.add_argument(
            '--gpu', default='0', type=str, action='store')
    parser.add_argument(
            '--tv_wd', default=1e-4, type=float, action='store')
    parser.add_argument(
            '--wd', default=1e-4, type=float, action='store')
    parser.add_argument(
            '--lr', default=0.05, type=float, action='store')
    parser.add_argument(
            '--global_start_idx', default=None, type=int, action='store')
    parser.add_argument(
            '--num_batches', default=4, type=int, action='store')
    parser.add_argument(
            '--layer_start_idx', default=None, type=int, action='store')
    parser.add_argument(
            '--layer_len_idx', default=1, type=int, action='store')
    parser = bs_fit.add_load_settings(parser)
    parser = bs_fit.add_model_settings(parser)
    return parser


class RawPtStimuliCompute(pt_stimuli_compute.PtStimuliCompute):
    def __init__(self, args, start_idx=0, batch_size=16):
        self.args = args
        self.start_idx = start_idx
        self.batch_size = batch_size
        self.device = torch.device("cuda")
        self.build_model()
        pass

    def build_input_images(self):
        args = self.args
        input_resolution = 224
        num_of_images = self.batch_size
        image_shape = (
                num_of_images,
                input_resolution, input_resolution, 
                3)
        self.images_var = torch.empty(*image_shape).to(
                self.device, torch.float).requires_grad_()

    def get_normalized_image(self):
        image = self.images_var.permute(0, 3, 1, 2)
        image = image + 0.5
        image = torch.clamp(image, min=0.0, max=1.0)
        image = self.xforms_augmentation(image)
        image = (image - self._mean) / self._std
        return image

    def xforms_augmentation(self, images):
        import pt_xforms as xforms
        scales = [1 + (i - 5) / 50. for i in range(11)]
        angles = list(range(-10, 11)) + 5 * [0]
        images = xforms.pad(images, 12)
        images = xforms.jitter(images, 8)
        images = xforms.random_scale(images, scales)
        images = xforms.random_rotate(images, angles)
        images = xforms.jitter(images, 4)
        return images

    def _build_mask_predictor(self):
        pass

    def clean_ops(self):
        pass

    def get_predictions(self, curr_output):
        predictions = curr_output.mean(dim=(2, 3))
        predictions = predictions[:, self.start_idx : self.start_idx + self.batch_size]
        return predictions


def _rfft2d_freqs(h, w):
    """Compute 2d spectrum frequences."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[:w//2+2]
    else:
        fx = np.fft.fftfreq(w)[:w//2+1]
    return np.sqrt(fx*fx + fy*fy)


color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_mean = [0.48, 0.46, 0.41]       


def _linear_decorelate_color(t):
    """Multiply input by sqrt of emperical (ImageNet) color correlation matrix.

    If you interpret t's innermost dimension as describing colors in a
    decorrelated version of the color space (which is a very natural way to
    describe colors -- see discussion in Feature Visualization article) the way
    to map back to normal colors is multiply the square root of your color
    correlations. 
    """
    # check that inner dimension is 3?
    t_flat = torch.reshape(t, [-1, 3])
    color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
    t_flat = torch.matmul(
            t_flat, 
            torch.from_numpy(color_correlation_normalized.T).to(
                torch.device("cuda"), torch.float))
    t = torch.reshape(t_flat, t.shape)
    return t


class FFTRawPtStimuliCompute(RawPtStimuliCompute):
    def __init__(self, args, start_idx=0, batch_size=16):
        self.args = args
        self.start_idx = start_idx
        self.batch_size = batch_size
        self.device = torch.device("cuda")
        self.build_model()
        pass

    def get_optimizer(self):
        args = self.args
        optimizer = torch.optim.Adam(
                [self.images_var], lr=args.lr)
        return optimizer

    def get_loss(self, predictions):
        loss = -predictions.trace()
        return loss

    def __initialize_images(self):
        sd = 0.01
        nn.init.normal_(self.images_var, 0, sd)

    def build_input_images(self):
        args = self.args
        self.input_resolution = 224
        num_of_images = self.batch_size

        self.freqs = _rfft2d_freqs(self.input_resolution, self.input_resolution)
        fh, fw = self.freqs.shape
        sd = 0.01
        image_shape = (num_of_images, 3, fh, fw, 2)
        self.images_var = torch.empty(*image_shape).to(
                self.device, torch.float).requires_grad_()

    def get_normalized_image(self):
        image = self.get_rgb_images_from_var()
        image = torch.clamp(image, min=0.0, max=1.0)
        image = self.xforms_augmentation(image)
        image = (image - self._mean) / self._std
        return image

    def get_uint_scale_image(self):
        image = self.get_rgb_images_from_var() * 255
        image = torch.clamp(image, min=0.0, max=255.0)
        return image

    def get_rgb_images_from_var(self):
        _images = []
        spertum_scale = 1.0 / np.maximum(
                self.freqs, 1.0/self.input_resolution)
        spertum_scale *= self.input_resolution
        spertum_scale = torch.from_numpy(spertum_scale[:, :, np.newaxis]).to(
                self.device, torch.float)

        for b_idx in range(self.batch_size):
            _image = []
            for c_idx in range(3):
                spectrum = self.images_var[b_idx, c_idx]
                scaled_spectrum = spectrum * spertum_scale
                _part_img = torch.irfft(scaled_spectrum, signal_ndim=2)
                _part_img = _part_img[:self.input_resolution, \
                                      :self.input_resolution]
                _image.append(_part_img)
            _image = torch.stack(_image, axis=2) / 4.
            _image = _linear_decorelate_color(_image)
            _image = _image.permute(2, 0, 1)
            _image = torch.sigmoid(_image)
            _images.append(_image)
        _images = torch.stack(_images, axis=0)
        return _images


def dump_result(save_result, model_id, layer, global_start_idx=None):
    #special = ''
    #special = '_xforms'
    special = '_lucid'
    if global_start_idx is not None:
        special = special + '_' + str(global_start_idx)
    result_path = raw_stimuli_compute.RESULT_PATH_PATTERN.format(
            model_id = model_id,
            layer = layer,
            special = special,
            )
    save_dir = os.path.dirname(result_path)
    if not os.path.isdir(save_dir):
        os.system('mkdir -p ' + save_dir)
    pickle.dump(save_result, open(result_path, 'wb'))


def main():
    parser = get_raw_pt_stimuli_compute_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.set_func, "Must specify set_func"
    args = bs_fit.load_set_func(args)

    batch_size = 16
    num_batches = args.num_batches
    #batch_size = 4
    #num_batches = 16
    #for layer in tqdm(['layer2.1.relu', 'layer4.1.relu']):
    layer_list = bs_fit.PT_RES18_LAYERS
    if args.layer_start_idx is not None:
        layer_end_idx = args.layer_start_idx + args.layer_len_idx
        layer_list = layer_list[\
                args.layer_start_idx : layer_end_idx]
    for layer in tqdm(layer_list):
        all_save_results = []
        for start_idx in range(0, num_batches * batch_size, batch_size):
            #stimuli_compute = RawPtStimuliCompute(args, start_idx, batch_size)
            stimuli_compute = FFTRawPtStimuliCompute(
                    args, 
                    start_idx + (args.global_start_idx or 0), 
                    batch_size)
            save_result = stimuli_compute.compute_optimal_stimuli(layer)
            all_save_results.append(save_result)
            del stimuli_compute

        images = []
        loss_vals = []
        actual_response = []

        for save_result in all_save_results:
            images.append(save_result['images'])
            loss_vals.append(save_result['loss_vals'])
            actual_response.append(save_result['actual_response'])

        images = np.concatenate(images, axis=0)
        loss_vals = np.sum(loss_vals, axis=0)
        actual_response = np.concatenate(actual_response, axis=0)
        final_save_result = {
                'images': images,
                'loss_vals': loss_vals,
                'actual_response': actual_response,
                }
        dump_result(
                final_save_result, args.identifier, 
                layer, args.global_start_idx)


if __name__ == '__main__':
    main()
