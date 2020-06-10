from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf

from tfutils import base

import json
import copy
import utils
import pdb

from cmd_parser import get_parser, load_setting
from param_setter import ParamSetter


def main():
    parser = get_parser()
    args = parser.parse_args()
    args = load_setting(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    param_setter = ParamSetter(args)
    params = param_setter.build_test_params()
    base.test_from_params(**params)


if __name__=='__main__':
    main()
