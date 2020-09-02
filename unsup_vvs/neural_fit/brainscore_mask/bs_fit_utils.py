import argparse
import copy
from argparse import Namespace
import tensorflow as tf
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import os
import sys
import importlib
import pdb
import numpy as np
import json
from collections import OrderedDict

import unsup_vvs.network_training.cmd_parser

# Color conversion code
def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()
    return out


def get_la_cmc_model(path):
    from unsup_vvs.neural_fit.pt_scripts.la_cmc_resnet import ResNetLabV1
    model = ResNetLabV1(skip_final_layer=True, before_pool=True)
    model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    checkpoint = torch.load(path, map_location=lambda storage, location: storage)
    model.load_state_dict(checkpoint['model_state_dict'])

    # freeze the layers
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def get_dc_model(path, verbose=True):
    import unsup_vvs.neural_fit.pt_scripts.resnet18_dc as resnet18_dc
    if verbose:
        print("=> loading checkpoint '{}'".format(path))
    checkpoint = torch.load(path, map_location=lambda storage, location: storage)

    # size of the top layer
    N = checkpoint['state_dict']['top_layer.bias'].size()

    # build skeleton of the model
    sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
    model = resnet18_dc.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

    # deal with a dataparallel table
    def rename_key(key):
        if not 'module' in key:
            return key
        return ''.join(key.split('.module'))

    checkpoint['state_dict'] = {rename_key(key): val
                                for key, val
                                in checkpoint['state_dict'].items()}

    # load weights
    model.load_state_dict(checkpoint['state_dict'])
    if verbose:
        print("Loaded")
    if torch.cuda.is_available():
        model.cuda()
        cudnn.benchmark = True

    # freeze the features layers
    for param in model.features.parameters():
        param.requires_grad = False
    model.eval()
    return model

        
class PtModelWithMid(nn.Module):
    """Creates logistic regression on top of frozen features for resnet"""
    def __init__(self, encoder, mid_layer):
        super(PtModelWithMid, self).__init__()
        self.encoder = encoder
        self.mid_layer = mid_layer

    def forward(self, x):
        x = self.encoder(x)
        return self.mid_layer(x)


def load_mid_layer(mid_ckpt_path):
    from analyze_scripts.pt_imagenet_transfer import RegLogMid
    mid_layer = RegLogMid()
    checkpoint = torch.load(mid_ckpt_path)
    mid_layer.load_state_dict(checkpoint['state_dict'])
    mid_layer.cuda()
    cudnn.benchmark = True

    # freeze the features layers
    for param in mid_layer.parameters():
        param.requires_grad = False
    mid_layer.eval()
    return mid_layer


def get_load_settings_from_func(args):
    train_args = cmd_parser.get_parser().parse_args([])
    train_args.load_setting_func = args.setting_name
    train_args = cmd_parser.load_setting(train_args)
    args.load_port = train_args.nport
    args.load_dbname = train_args.dbname
    args.load_colname = train_args.collname
    args.load_expId = train_args.expId
    return args


def load_set_func(args):
    all_paths = args.set_func.split('.')
    module_name = '.'.join(['settings'] + all_paths[:-1])
    load_setting_module = importlib.import_module(module_name)
    set_func = all_paths[-1]
    set_func = getattr(load_setting_module, set_func)
    args = set_func(args)
    if args.setting_name is not None and args.load_from_ckpt is None:
        args = get_load_settings_from_func(args)
    return args


def color_normalize(image):
    image = tf.cast(image, tf.float32) / 255
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - imagenet_mean) / imagenet_std
    return image
