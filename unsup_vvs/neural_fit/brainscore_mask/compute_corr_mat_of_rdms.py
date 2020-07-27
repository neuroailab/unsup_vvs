import pickle
import argparse
import os
import sys
import pdb
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
RESULT_CACHING_DIR = '/mnt/fs4/chengxuz/v4it_temp_results/.result_caching'
DEFAULT_SAVE_DIR = os.path.join(RESULT_CACHING_DIR, 'computed_rdms')
ACTIVATION_PATTERN = 'activations'
TF_MODEL_LAYERS = ['encode_%i' % i for i in range(1, 10)]
PT_MODEL_LAYERS = ['maxpool'] \
        + ['layer1.0.relu', 'layer1.1.relu'] \
        + ['layer2.0.relu', 'layer2.1.relu'] \
        + ['layer3.0.relu', 'layer3.1.relu'] \
        + ['layer4.0.relu', 'layer4.1.relu']


def get_rdm_pkls():
    all_pkls = os.listdir(DEFAULT_SAVE_DIR)
    all_pkls = list(filter(lambda name: ACTIVATION_PATTERN in name, all_pkls))
    all_pkls = sorted(all_pkls)
    all_pkls = [os.path.join(DEFAULT_SAVE_DIR, each_pkl) for each_pkl in all_pkls]
    return all_pkls


def get_lower_triangle(mat):
    n, m = mat.shape
    indexs = np.tril_indices(n=n, m=m, k=-1)
    mat = mat[indexs]
    return mat


def main():
    all_rdm_paths = get_rdm_pkls()
    all_rdms = [pickle.load(open(_path, 'rb')) for _path in tqdm(all_rdm_paths)]

    num_of_rdms = len(all_rdm_paths)
    corr_mat = np.zeros([num_of_rdms*9, num_of_rdms*9])
    for _idx_1 in tqdm(range(num_of_rdms)):
        for _idx_2 in range(_idx_1, num_of_rdms):
            _rdms_1 = all_rdms[_idx_1]
            _layers_1 = TF_MODEL_LAYERS
            if _layers_1[0] not in _rdms_1:
                _layers_1 = PT_MODEL_LAYERS

            _rdms_2 = all_rdms[_idx_2]
            _layers_2 = TF_MODEL_LAYERS
            if _layers_2[0] not in _rdms_2:
                _layers_2 = PT_MODEL_LAYERS

            for _layer_idx_1, _layer_1 in enumerate(_layers_1):
                for _layer_idx_2, _layer_2 in enumerate(_layers_2):
                    _final_idx_1 = 9 * _idx_1 + _layer_idx_1
                    _final_idx_2 = 9 * _idx_2 + _layer_idx_2
                    _rdm_1_lt = get_lower_triangle(_rdms_1[_layer_1])
                    _rdm_2_lt = get_lower_triangle(_rdms_2[_layer_2])
                    corr_mat[_final_idx_1, _final_idx_2] = pearsonr(
                            _rdm_1_lt, _rdm_2_lt)[0]
    save_results = {
            'corr_mat': corr_mat,
            'all_rdm_paths': all_rdm_paths,
            }
    pickle.dump(
            save_results, 
            open(os.path.join(DEFAULT_SAVE_DIR, 'corr_mat.pkl'), 'wb'))


if __name__ == '__main__':
    main()
