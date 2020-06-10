import argparse
from nf_cmd_parser import add_general_settings, add_network_cfg_settings


def add_brainscore_settings(parser):
    parser.add_argument(
            '--batchsize', default=64, type=int, action='store',
            help='Batch size')
    parser.add_argument(
            '--benchmark', default=None, type=str, action='store',
            help='Benchmark name from brainscore')
    return parser


def add_model_settings(parser):
    parser.add_argument(
            '--model_type', default='vm_model', type=str, action='store')
    parser.add_argument(
            '--prep_type', default='mean_std', type=str, action='store')
    parser.add_argument(
            '--setting_name', default=None, type=str, action='store',
            help='Network setting name')
    return parser


def add_data_settings(parser):
    parser.add_argument(
            '--data_norm_type', default='standard', type=str, action='store')
    parser.add_argument(
            '--img_out_size', default=224, type=int, action='store')
    return parser


def get_brainscore_parser():
    parser = argparse.ArgumentParser(
            description='The script to fit to neural data using brain score')
    parser = add_general_settings(parser)
    parser = add_network_cfg_settings(parser)
    parser = add_brainscore_settings(parser)
    parser = add_model_settings(parser)
    parser = add_data_settings(parser)
    return parser
