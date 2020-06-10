from __future__ import division, print_function, absolute_import

import pymongo as pm
import gridfs
from tensorflow.core.protobuf import saver_pb2
import tarfile
import cPickle

import numpy as np
from scipy import misc
import os
import time
import sklearn.linear_model
import math

import tensorflow as tf
from tfutils.db_interface import verify_pb2_v2_files
from model.instance_model import resnet_embedding
from utils import DATA_LEN_IMAGENET_FULL


def _print_checkpt_vars(path):
    # For debugging
    from tensorflow.python.tools.inspect_checkpoint import (
        print_tensors_in_checkpoint_file
    )
    print_tensors_in_checkpoint_file(path,
                                     all_tensor_names=True,
                                     all_tensors=False,
                                     tensor_name='')


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

        fs_file = open(path, 'wrb+')
        self.fs_bucket.download_to_stream(rec['_id'], fs_file)
        fs_file.close()

        if rec['_saver_write_version'] == saver_pb2.SaverDef.V2:
            assert fname.endswith('.tar')
            tar = tarfile.open(path)
            tar.extractall(path=self.load_files_dir)
            tar.close()
            path = os.path.splitext(path)[0]
            verify_pb2_v2_files(path, rec)
        return path


def load_model(sess, tfutils_reader, step_num=None,
               load_mem_bank=False,
               mem_bank_shape=None,
               input_dtype=tf.uint8,
               # Optionally pass in your own input_handle
               input_handle=None):
    q = { 'saved_filters': True }
    if step_num is not None:
        q['step'] = { '$lte': step_num }
    recs = tfutils_reader.query(q, sort=[('step', -1)])
    rec = recs[0]

    if step_num is not None and rec['step'] != step_num:
        print('Could not find record at step %r, using step %r instead.'
              % (step_num, rec['step']))

    restore_path = tfutils_reader.load_gridfs_file(rec)
    print('Restoring model from', restore_path)

    existing_vars = set(tf.global_variables())
    labels, memory_bank = None, None
    if load_mem_bank:
        data_len, embedding_dim = mem_bank_shape
        with tf.variable_scope("instance", reuse=tf.AUTO_REUSE):
            memory_bank = tf.get_variable(
                'memory_bank', shape=(data_len, embedding_dim),
                dtype=tf.float32, initializer=tf.zeros_initializer,
                trainable=False)
            labels = tf.get_variable(
                'all_labels', shape=(data_len),
                dtype=tf.int64, initializer=tf.zeros_initializer,
                trainable=False)

    # TODO: allow different params for resnet (or eventually other embeddings)
    crop_size = 224
    if input_handle is None:
        input_handle = tf.placeholder(dtype=input_dtype, shape=(None, 224, 224, 3))
    else:
        assert tuple(input_handle.get_shape().as_list()[1:]) == (224, 224, 3)
    embedding = resnet_embedding(input_handle)

    restore_vars = {}
    for var in tf.global_variables():
        if var is input_handle or var in existing_vars:
            continue
        scope_prefix = tf.get_variable_scope().name
        if scope_prefix != '':
            scope_prefix += '/'
        vname = var.op.name
        if not vname.startswith(scope_prefix):
            continue
        # checkpoint variables do not have the scope prefix
        plain_name = vname[len(scope_prefix):]
        restore_vars[plain_name] = var
    saver = tf.train.Saver(restore_vars)
    saver.restore(sess, restore_path)
    return {
        "embedding": embedding, "input_handle": input_handle,
        "memory_bank": memory_bank, "labels": labels
    }


def load_validation_results(tfutils_reader, validation_keys,
                            steps_per_epoch, epoch_range=None):
    q = {'validation_results': {'$exists': True}}
    if epoch_range is not None:
        start, end = epoch_range
        q['step'] = {
            '$gte': start * steps_per_epoch, '$lte': end * steps_per_epoch
        }
    recs = tfutils_reader.query(
        q, restrict_fields={'step': 1, 'validation_results': 1})
    # Have to sort ourselves rather than through mongo query because
    # for some reason mongodb hits a memory limit even for very small
    # results. Can possibly be fixed by adding an index for 'step'.
    def sort_fn(x):
        gen_time = x['_id'].generation_time
        t = time.mktime(gen_time.timetuple())
        return x['step'], -t
    recs = sorted(recs, key=sort_fn)

    def is_matching_rec(r):
        x = r['validation_results']
        for key in validation_keys:
            if key not in x:
                return False
            x = x[key]
        return True

    ret = []
    for r in recs:
        if not is_matching_rec(r):
            continue
        if len(ret) > 0 and ret[-1][0] == r['step']:
            continue
        value = r['validation_results']
        for key in validation_keys:
            value = value[key]
        ret.append((r['step'], value))
    return [(s/steps_per_epoch, val) for s, val in ret]


def load_loss_results(tfutils_reader, key,
                      conv_len=100):
    train_entries = tfutils_reader.query({
        'train_results': {'$exists': True}
        }, projection=['train_results', 'step'])
    train_entries = sorted(train_entries, key=lambda x: x['step'])

    last_step = -1
    losses = []
    for entry in train_entries:
        step = entry['step']
        if step == last_step:
            continue

        last_step = step
        for idx, results in enumerate(entry['train_results']):
            # step + idx
            losses.append(results[key])

    # Smooth the loss curve
    losses = np.array(losses)
    conv_list = np.ones([conv_len])/conv_len
    losses = np.convolve(losses, conv_list, mode='valid')
    return losses


if __name__ == '__main__':
    # # Example code below of how to use various functions defined here

    # tr = TfutilsReader('instance_task', 'control', 'full', port=27009,
    #                    cache_dir='/mnt/fs1/azhai/checkpoints')
    # results = load_validation_results(tr, ['topn', 'top1'], 10009)
    # for epoch, value in results:
    #     print(epoch, value)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # tr = TfutilsReader('instance_task', 'control', 'full', port=27009,
    #                    cache_dir='/mnt/fs1/azhai/checkpoints')
    # model_data = load_model(sess, tr, step_num=20000, load_mem_bank=True)
    # print(model_data)
    # # TODO: actually evaluate something
    # sess.close()
    pass
