"""
Requires python 3.6 and streams repo installed
"""
import h5py
import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import PCA
import pickle
import os, sys


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to compute i2n, o1, o2 metric')

    parser.add_argument(
            '--hdf5_path', 
            default='/home/chengxuz/instance_on_objectome_again.hdf5', 
            type=str, action='store',
            help='Path to the responses hdf5')
    parser.add_argument(
            '--subsample_method', 
            default='average', 
            type=str, action='store',
            help='Method to subsample neural responses')
    parser.add_argument(
            '--save_path', 
            default='/mnt/fs4/chengxuz/v4it_temp_results/inst_res18.pkl', 
            type=str, action='store',
            help='Path to the saved results')
    return parser


def get_subsample_feats(model_feat, args):
    if args.subsample_method=='average':
        if len(model_feat.shape) == 4:
            model_feat = np.mean(model_feat, axis=1)
            model_feat = np.mean(model_feat, axis=1)

    elif args.subsample_method=='subsample':
        print('Subsampling 1000 units')
        model_feat = model_feat.reshape([2400, -1])
        # subsample 1000 units
        choose_idxs = np.random.choice(model_feat.shape[1], 1000)
        model_feat = model_feat[:, choose_idxs]

    elif args.subsample_method=='center':
        print('Subsampling central units')
        model_feat = model_feat[:, 3:6, 3:6, :]
        model_feat = model_feat.reshape([2400, -1])

    elif args.subsample_method=='PCA':
        model_feat = model_feat.reshape([2400, -1])
        curr_num_units = model_feat.shape[1]
        target_num_units = 256
        if curr_num_units > target_num_units:
            PCA_fitter = PCA(n_components=target_num_units)
            model_feat = PCA_fitter.fit_transform(model_feat)

    elif args.subsample_method=='MAXMIN':
        if len(model_feat.shape) == 4:
            model_feat_ave = np.mean(model_feat, axis=1)
            model_feat_ave = np.mean(model_feat_ave, axis=1)
            model_feat_max = model_feat.max(axis=1).max(axis=1)
            model_feat_min = model_feat.min(axis=1).min(axis=1)
            model_feat_std = np.std(model_feat, axis=(1, 2))
            model_feat = np.concatenate(
                    [model_feat_ave, model_feat_max, 
                     model_feat_min, model_feat_std],
                    axis=-1)

    elif args.subsample_method=='None':
        pass
    else:
        raise NotImplementedError('Other subsample methods not supported!')

    return model_feat


def reorder_feats(model_feat):
    # Change the order
    data = pd.read_pickle('/mnt/fs4/chengxuz/.streams/objectome/meta.pkl')
    new_order = list(data['id'])
    new_feat = np.zeros(model_feat.shape)
    arg_sort = [i[0] for i in sorted(enumerate(new_order), key=lambda x:x[1])]
    for old_pos, new_pos in enumerate(arg_sort):
        new_feat[new_pos] = model_feat[old_pos]
    return new_feat


def main():
    parser = get_parser()
    args = parser.parse_args()
    os.environ['STREAMS_ROOT'] = '/mnt/fs4/chengxuz/.streams'
    from streams.metrics.behav_cons import objectome_cons

    fin = h5py.File(args.hdf5_path, 'r')
    all_res = {}

    for which_layer in fin.keys():
        print(which_layer)
        model_feat = fin[which_layer][:]
        model_feat = get_subsample_feats(model_feat, args)
        new_feat = reorder_feats(model_feat)
        all_res[which_layer] = {}

        print('Finish the reordering')
        df = objectome_cons(new_feat)
        print(list(df['r']))
        all_res[which_layer]['i2n'] = list(df['r'])

        df = objectome_cons(new_feat, metric='o1')
        all_res[which_layer]['o1'] = list(df['r'])

        df = objectome_cons(new_feat, metric='o2')
        all_res[which_layer]['o2'] = list(df['r'])

    pickle.dump(all_res, open(args.save_path, 'wb'), protocol=2)

if __name__ == '__main__':
    main()
