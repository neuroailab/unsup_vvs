import pickle
import argparse
import os
import sys
import pdb
from tqdm import tqdm
import numpy as np
RESULT_CACHING_DIR = '/mnt/fs4/chengxuz/v4it_temp_results/.result_caching'
DEFAULT_SAVE_DIR = os.path.join(RESULT_CACHING_DIR, 'computed_rdms')
ACTIVATION_DIR = os.path.join(
        RESULT_CACHING_DIR,
        'model_tools.activations.core.ActivationsExtractorHelper._from_paths_stored')
ACTIVATION_PATTERN = 'activations'


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to compute RDMs from activations')
    parser.add_argument(
            '--save_dir', type=str, 
            default=DEFAULT_SAVE_DIR,
            action='store',
            help='Directory for saving rdm results')
    return parser


def get_activation_pkls():
    all_pkls = os.listdir(ACTIVATION_DIR)
    all_pkls = list(filter(lambda name: ACTIVATION_PATTERN in name, all_pkls))
    all_pkls = sorted(all_pkls)
    all_pkls = [os.path.join(ACTIVATION_DIR, each_pkl) for each_pkl in all_pkls]
    return all_pkls


def main():
    parser = get_parser()
    args = parser.parse_args()
    all_pkls = get_activation_pkls()

    os.system('mkdir -p ' + args.save_dir)
    for each_pkl in tqdm(all_pkls):
        save_path = os.path.join(
                args.save_dir,
                os.path.basename(each_pkl))
        if os.path.exists(save_path):
            continue

        activations = pickle.load(open(each_pkl, 'rb'))['data']
        all_layers = np.unique(activations.layer)
        act_arr = np.asarray(activations)
        layer_names = np.asarray(activations.layer)

        _rdms = {}
        for each_layer in all_layers:
            _resp = act_arr[:, layer_names == each_layer]
            _rdms[each_layer] = np.corrcoef(_resp)
        pickle.dump(_rdms, open(save_path, 'wb'))


if __name__ == '__main__':
    main()
