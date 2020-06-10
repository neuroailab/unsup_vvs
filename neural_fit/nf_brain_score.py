"""
Using brain-score interface to do the neural fitting
requiring python 3.6 and brain-score.
"""

import pdb
from tqdm import tqdm
import brainscore
import numpy as np
import h5py
import argparse
import os
import sys
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr

TRAIN_NUM = 5500
SEED = 0
from brainscore.model_interface import BrainModel
from brainio_base.assemblies import NeuroidAssembly


class PrecomputedFeatures(BrainModel):
    def __init__(self, features):
        self.features = features

    def start_recording(self, region, *args, **kwargs):
        pass

    def look_at(self, stimuli):
        assert set(self.features['image_id'].values) == set(stimuli['image_id'].values)
        features = self.features.isel(presentation=[np.where(self.features['image_id'].values == image_id)[0][0]
                                                    for image_id in stimuli['image_id'].values])
        assert all(features['image_id'].values == stimuli['image_id'].values)
        return self.features


def get_parser():
    parser = argparse.ArgumentParser(
            description='Using brain-score interface to do the neural fitting')

    parser.add_argument(
            '--hdf5_path', 
            default='/mnt/fs4/chengxuz/v1_cadena_related/results/cate.hdf5', 
            type=str, action='store',
            help='Hdf5 file containing the results')
    return parser


def get_v1_cadena_data():
    from brainscore.assemblies.private import ToliasCadena2017Loader
    cadena_loader = ToliasCadena2017Loader()
    data = cadena_loader()
    data = data.sortby(data.image_file_name)
    return data


def build_pls_wrapper(curr_data, neural_df):
    assert len(curr_data.shape) == 2, "Must be 2-dim array"
    coords = {'neuroid_id': ('neuroid', np.arange(curr_data.shape[1]))}
    for presentation_key in neural_df.stimulus_set.keys():
        if getattr(neural_df, presentation_key, None) is not None:
            coords[presentation_key] = ('presentation', 
                                        neural_df.coords[presentation_key])
    curr_assembly = NeuroidAssembly(
            curr_data,
            coords=coords,
            dims=['presentation', 'neuroid'])
    model_wrapper = PrecomputedFeatures(curr_assembly)
    return model_wrapper


def build_wrapper(curr_data, neural_df):
    curr_data = np.transpose(curr_data, [0, 3, 1, 2])
    channel_dim = curr_data.shape[1]
    channel_x_dim = curr_data.shape[2]
    channel_y_dim = curr_data.shape[3]
    channel_coord = np.tile(
            np.arange(channel_dim)[:, np.newaxis, np.newaxis],
            [1, channel_x_dim, channel_y_dim])
    channel_x_coord = np.tile(
            np.arange(channel_x_dim)[np.newaxis, :, np.newaxis],
            [channel_dim, 1, channel_y_dim])
    channel_y_coord = np.tile(
            np.arange(channel_y_dim)[np.newaxis, np.newaxis, :],
            [channel_dim, channel_x_dim, 1])

    curr_data = curr_data.reshape((curr_data.shape[0], -1))
    coords = {
            'neuroid_id': ('neuroid', np.arange(curr_data.shape[1])),
            'channel': ('neuroid', channel_coord.reshape([-1])),
            'channel_x': ('neuroid', channel_x_coord.reshape([-1])),
            'channel_y': ('neuroid', channel_y_coord.reshape([-1]))}
    for presentation_key in neural_df.stimulus_set.keys():
        if getattr(neural_df, presentation_key, None) is not None:
            coords[presentation_key] = ('presentation', 
                                        neural_df.coords[presentation_key])
    curr_assembly = NeuroidAssembly(
            curr_data,
            coords=coords,
            dims=['presentation', 'neuroid'])
    model_wrapper = PrecomputedFeatures(curr_assembly)
    return model_wrapper


def main():
    from brainscore.benchmarks.neural import \
            ToliasCadena2017PLS, ToliasCadena2017Mask
    parser = get_parser()
    args = parser.parse_args()

    v1_cadena_data = get_v1_cadena_data()
    network_data_fin = h5py.File(args.hdf5_path, 'r')
    v1_cadena_benchmark = ToliasCadena2017PLS()
    #v1_cadena_benchmark = ToliasCadena2017Mask()

    for each_layer in network_data_fin.keys():
        curr_data = np.asarray(network_data_fin[each_layer])
        model_wrapper = build_wrapper(curr_data, v1_cadena_data)
        score = v1_cadena_benchmark(model_wrapper)
        print(each_layer)
        print(np.asarray(score))
        #pdb.set_trace()


if __name__ == '__main__':
    main()
