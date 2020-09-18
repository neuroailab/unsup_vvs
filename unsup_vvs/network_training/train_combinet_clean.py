from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf

from tfutils import base

import json
import copy
import utils
import pdb

from unsup_vvs.network_training.cmd_parser import get_parser, load_setting
from unsup_vvs.network_training.param_setter import ParamSetter


def main():
    parser = get_parser()
    args = parser.parse_args()
    args = load_setting(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if len(args.innerargs)==0:
        param_setter = ParamSetter(args)
        params = param_setter.build_all_params()
        base.train_from_params(**params)
    else:
        params = {
                'save_params': [],
                'load_params': [],
                'model_params': [],
                'train_params': None,
                'loss_params': [],
                'learning_rate_params': [],
                'optimizer_params': [],
                'log_device_placement': False,
                'validation_params': [],
                'skip_check': True,
                }

        list_names = [
                "save_params", "load_params", "model_params",
                "validation_params", "loss_params", "learning_rate_params",
                "optimizer_params"
                ]

        for curr_arg in args.innerargs:
            args = parser.parse_args(curr_arg.split())
            args = load_setting(args)
            param_setter = ParamSetter(args)
            curr_params = param_setter.build_all_params()
            for tmp_key in list_names:
                params[tmp_key].append(curr_params[tmp_key])
            params['train_params'] = curr_params['train_params']
        base.train_from_params(**params)


if __name__=='__main__':
    main()
