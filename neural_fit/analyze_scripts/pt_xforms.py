from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import pdb


def pad(images, w=12):
    images = F.pad(images, value=0.5, pad=(w, w, w, w))
    return images


def jitter(images, d):
    curr_shape = images.shape
    sta_x = np.random.randint(d)
    sta_y = np.random.randint(d)
    images = images[
            :, :, \
            sta_x : sta_x + curr_shape[2] - d, \
            sta_y : sta_y + curr_shape[3] - d]
    return images


def random_scale(images, scales):
    _scale = np.random.choice(scales)
    images = F.interpolate(
            images, 
            scale_factor=[1, 1, _scale, _scale],
            mode='bilinear')
    return images


def random_rotate(images, degrees):
    # Require https://github.com/neuroailab/torchsample.git
    from torchsample.transforms import Rotate
    _degree = np.random.choice(degrees)
    rotate_func = Rotate(_degree)

    ret_images = []
    for _image in images:
        _image = rotate_func(_image)
        ret_images.append(_image)
    return torch.stack(ret_images)
