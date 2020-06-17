from __future__ import division, print_function, absolute_import

import pymongo as pm
import gridfs
from tensorflow.core.protobuf import saver_pb2
import tarfile

import numpy as np
from scipy import misc
import os
import time
import sklearn.linear_model
import math

import tensorflow as tf

DEFAULT_MODEL_CACHE_DIR = '/data5/chengxuz/Dataset/unsup_vvs_datasets/brainscore_model_caches'


class TfutilsReader(object):
    def __init__(self, dbname, colname, exp_id,
                 port, cache_dir):
        self.exp_id = exp_id
        self.conn = conn = pm.MongoClient(port=port)

        self.coll = conn[dbname][colname + '.files']
        self.collfs = gridfs.GridFS(conn[dbname], colname)
        self.fs_bucket = gridfs.GridFSBucket(conn[dbname], colname)

        self.load_files_dir = os.path.join(cache_dir, dbname, colname, exp_id)

    def query(self, query_dict, restrict_fields=None, **kwargs):
        # commonly used kwargs: sort, projection
        query_dict = query_dict.copy()
        query_dict['exp_id'] = self.exp_id
        if restrict_fields is None:
            return self.coll.find(query_dict, **kwargs)
        return self.coll.find(query_dict, restrict_fields, **kwargs)

    def load_gridfs_file(self, rec):
        '''
        Converts a GridFS file to an ordinary file and returns the
        path where the GridFS contents were copied.
        '''
        assert 'saved_filters' in rec

        if not os.path.exists(self.load_files_dir):
            os.makedirs(self.load_files_dir)
        fname = os.path.basename(rec['filename'])
        path = os.path.join(self.load_files_dir, fname)

        if rec['_saver_write_version'] == saver_pb2.SaverDef.V2:
            extracted_path = os.path.splitext(path)[0]
            if os.path.exists(extracted_path + '.index'):
                print('Using already present file at extraction path %s.'
                      % extracted_path)
                return extracted_path
        elif os.path.exists(path):
            print('Using already present file at extraction path %s.' % path)
            return path

        fs_file = open(path, 'wb+')
        self.fs_bucket.download_to_stream(rec['_id'], fs_file)
        fs_file.close()

        if rec['_saver_write_version'] == saver_pb2.SaverDef.V2:
            assert fname.endswith('.tar')
            tar = tarfile.open(path)
            tar.extractall(path=self.load_files_dir)
            tar.close()
            os.system('rm ' + path)
            path = os.path.splitext(path)[0]
            from tfutils.db_interface import verify_pb2_v2_files
            verify_pb2_v2_files(path, rec)
        return path


def load_model_from_mgdb(db, col, exp, port, cache_dir, step_num):
    tfutils_reader = TfutilsReader(
        db, col, exp, 
        port=port, cache_dir=cache_dir)
    q = { 'saved_filters': True }
    q['step'] = { '$lte': step_num }
    recs = tfutils_reader.query(q, sort=[('step', -1)])
    rec = recs[0]
    if step_num is not None and rec['step'] != step_num:
        print('Could not find record at step %r, using step %r instead.'
              % (step_num, rec['step']))
    restore_path = tfutils_reader.load_gridfs_file(rec)
    return restore_path


if __name__ == '__main__':
    #print(load_model_from_mgdb(
    #        'irla_and_others', 
    #        'res18', 
    #        'la', 
    #        27007, 
    #        DEFAULT_MODEL_CACHE_DIR, 
    #        step_num=300*10009))
    #print(load_model_from_mgdb(
    #        'cate_aug', 
    #        'res18', 
    #        'exp_seed0_bn_wd', 
    #        27007, 
    #        DEFAULT_MODEL_CACHE_DIR, 
    #        step_num=300*10009))
    #print(load_model_from_mgdb(
    #        'vd_unsup_fx', 
    #        'infant', 
    #        'vd_3dresnet_full_v2', 
    #        27007, 
    #        DEFAULT_MODEL_CACHE_DIR, 
    #        step_num=500*10009))
    print(load_model_from_mgdb(
            'vd_unsup_fx', 
            'infant', 
            'vd_ctl_infant', 
            27007, 
            DEFAULT_MODEL_CACHE_DIR, 
            step_num=500*10009))
