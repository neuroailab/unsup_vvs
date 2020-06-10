from __future__ import division, print_function, absolute_import
import cPickle
import os, sys
import numpy as np
import json
import copy
import argparse
import h5py
import time
import pdb
import tensorflow as tf

from sklearn.decomposition import PCA

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_parser():
    parser = argparse.ArgumentParser(description='The script to compute the stats and do the PCA')

    # General settings
    parser.add_argument('--hdf5path', 
            default='/data2/chengxuz/vm_response/response_f2bpslen.hdf5,'+
            '/data2/chengxuz/vm_response/response_f2bpsldec.hdf5',
            type=str, action='store',
            help='Path to get the exported features (multiple files could be split by ",")')
    parser.add_argument('--hdf5path_val', 
            default='/data2/chengxuz/vm_response/response_f2bpslen_val.hdf5,'+
            '/data2/chengxuz/vm_response/response_f2bpsldec_val.hdf5', 
            type=str, action='store',
            help='Path of features on validation dataset')
    parser.add_argument('--PCApath', 
            default='/data2/chengxuz/vm_response/save_PCAs/',
            type=str, action='store',
            help='Path for saving computed PCA object for each key')
    parser.add_argument('--keys', default='encode_17:decode_1',
            type=str, action='store',
            help='All keys needed, split by ",", concatenation indicated by ":"')
    parser.add_argument('--tfrpath', default='/data2/chengxuz/vm_response/tfrecords/',
            type=str, action='store',
            help='Path to store the tfrecords')
    parser.add_argument('--num_comp', default='512',
            type=str, action='store',
            help='Number of components for each key')
    parser.add_argument('--num_per_tfr', default=256,
            type=int, action='store',
            help='Number of records for each tfrecord file')

    return parser

def load_from_hdf5s(each_key, fin_list):
    curr_key_list = each_key.split(':')
    curr_feat = None
    for curr_key in curr_key_list:
        found = False
        for each_fin in fin_list:
            if curr_key in each_fin:
                found = True
                new_feat = np.asarray(each_fin[curr_key])
                #new_feat = np.asarray(each_fin[curr_key][:256])
                if curr_feat is None:
                    curr_feat = new_feat
                else:
                    #print(curr_feat.shape)
                    #print(new_feat.shape)
                    #assert curr_feat.shape[:-1]==new_feat[:-1], "All but last dimensions must match"
                    curr_feat = np.concatenate([curr_feat, new_feat], axis=-1)
                break
        assert found, "Feature %s not found!" % curr_key
    print(curr_feat.shape)
    #pdb.set_trace()
    space_shape = curr_feat.shape[1]
    curr_feat = curr_feat.reshape([-1,curr_feat.shape[-1]])

    return space_shape,curr_feat

def get_fin_list(hdf5path):
    all_hdf5path = hdf5path.split(',')
    fin_list = []
    for each_hdf5path in all_hdf5path:
        temp_fin = h5py.File(each_hdf5path, 'r')
        fin_list.append(temp_fin)

    return fin_list

def write_to_tfrs(new_feat, tfrpath, each_key, num_per_tfr, name_pat='train_%i.tfrecords'):
    each_key = each_key.replace(':','-')
    new_path = os.path.join(tfrpath, each_key)
    os.system('mkdir -p %s' % new_path)

    data_len = new_feat.shape[0]
    for file_idx in xrange(int(data_len/num_per_tfr)):
        curr_filepath = os.path.join(new_path, name_pat % file_idx)

        writer = tf.python_io.TFRecordWriter(curr_filepath)
        for indx_now in xrange(file_idx*num_per_tfr, (file_idx+1)*num_per_tfr):
            img = new_feat[indx_now]
            img_raw = img.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                each_key: _bytes_feature(img_raw)}))
            writer.write(example.SerializeToString())
        writer.close()

    return new_path,each_key
    
def main():
    parser = get_parser()
    args = parser.parse_args()

    key_list = args.keys.split(',')
    nc_list = args.num_comp.split(',')
    if len(nc_list)<len(key_list):
        curr_len = len(nc_list)
        for _ in range(curr_len, len(len(key_list))):
            nc_list.append(nc_list[-1])
    nc_list = [int(each_nc) for each_nc in nc_list]
    fin_list = get_fin_list(args.hdf5path)
    fin_list_val = get_fin_list(args.hdf5path_val)

    ret_dict = {}

    for each_key,each_nc in zip(key_list,nc_list):
        # For each key, find and concatenate all subkeys,
        # compute the stats, and do the PCA
        space_shape,curr_feat = load_from_hdf5s(each_key, fin_list)
        ## Normalize the data for each channel
        mean_feat = np.mean(curr_feat, axis=0)
        std_feat = np.std(curr_feat, axis=0)
        curr_feat = (curr_feat - mean_feat)/std_feat
        curr_PCA = PCA(n_components=each_nc)
        print('------Begin PCA fit------')
        curr_PCA.fit(curr_feat)
        new_feat = curr_PCA.transform(curr_feat)
        new_shape = [-1,space_shape,space_shape,new_feat.shape[-1]]
        new_feat = new_feat.reshape(new_shape)
        write_to_tfrs(new_feat, args.tfrpath, each_key, args.num_per_tfr, name_pat='train_%i.tfrecords')

        # Get feat from validation and transform it
        _,curr_feat_val = load_from_hdf5s(each_key, fin_list_val)
        curr_feat_val = (curr_feat_val - mean_feat)/std_feat
        new_feat_val = curr_PCA.transform(curr_feat_val)
        new_feat_val = new_feat_val.reshape(new_shape)
        print(new_feat_val.shape)
        tfrsdir_path, new_name = write_to_tfrs(
                new_feat_val, args.tfrpath, each_key, 
                args.num_per_tfr, name_pat='test_%i.tfrecords')

        curr_meta = {new_name:{'shape': (), 'dtype': tf.string}}
        cPickle.dump(curr_meta, open(os.path.join(tfrsdir_path, 'meta.pkl'), 'w'))

        curr_PCA_path = os.path.join(args.PCApath, '%s.pkl' % new_name)
        cPickle.dump({'PCA': curr_PCA, 'new_shape': new_shape}, open(curr_PCA_path, 'w'))

        #pdb.set_trace()

if __name__ == '__main__':
    main()
