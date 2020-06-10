from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import cPickle
import pdb

import json
import copy
import argparse
import h5py
import time

def get_parser():
    parser = argparse.ArgumentParser(description='The script to combine the train and val data into one hdf5')

    # General settings
    parser.add_argument('--train_hdf5', default='/data2/chengxuz/vm_response/response_resnet34.hdf5', 
            type=str, action='store', help='Train hdf5')
    parser.add_argument('--test_hdf5', default='/data2/chengxuz/vm_response/response_resnet34_val.hdf5', 
            type=str, action='store', help='Test hdf5')
    parser.add_argument('--ave_pkl', default='/data2/chengxuz/vm_response/all_average_temp.pkl', 
            type=str, action='store', help='Neural responses index')
    parser.add_argument('--saving_hdf5', default='/data2/chengxuz/vm_response/response_resnet34_all.hdf5', 
            type=str, action='store', help='Hdf5 to save')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    fin_train = h5py.File(args.train_hdf5)
    fin_test = h5py.File(args.test_hdf5)
    fout = h5py.File(args.saving_hdf5, 'w')

    all_keys = fin_train.keys()

    all_average = cPickle.load(open(args.ave_pkl, 'r'))
    num_neurons = 2
    all_IT_2 = all_average[:,:num_neurons]

    #for each_key in all_keys:
    for each_key in ['encode_11', 'encode_5']:
        if each_key in ['IT_ave', 'V4_ave']:
            continue
        key_shape = fin_train[each_key].shape
        curr_save_shape = [5760] + list(key_shape[1:])
        curr_data = np.zeros(curr_save_shape, dtype=np.float32)

        curr_idxs = []
        
        for each_fin in [fin_train, fin_test]:
            curr_IT = each_fin['IT_ave']
            curr_IT_2 = curr_IT[:, :num_neurons]

            train_or_test_idxs = []
            for indx in xrange(curr_IT_2.shape[0]):
                all_flags = all_IT_2[:,0]==curr_IT[indx,0]
                for temp_idx in xrange(1, num_neurons):
                    all_flags = all_flags & (all_IT_2[:,temp_idx]==curr_IT[indx,temp_idx])
                find_0 = np.where(all_flags)
                if len(np.unique(find_0[0]))!=1:
                    print(find_0)
                    pdb.set_trace()
                curr_data[find_0[0][0]] = each_fin[each_key][indx]
                curr_idxs.append(find_0[0][0])
                train_or_test_idxs.append(find_0[0][0])
            print(len(np.unique(train_or_test_idxs)))
        print(len(np.unique(curr_idxs)))
        #pdb.set_trace()
         
        print('Finish key %s' % each_key)
        dataset_tmp = fout.create_dataset(each_key, data=curr_data)

if __name__ == '__main__':
    main()
