import cPickle
import os
import tensorflow as tf

dir_prefix = '/mnt/fs1/Dataset/neural_resp/V4IT'
meta_name = 'meta.pkl'

IT_meta = {'IT_ave': {'shape': (), 'dtype': tf.string}}
V4_meta = {'V4_ave': {'shape': (), 'dtype': tf.string}}

for curr_split in xrange(5):
    curr_dir = dir_prefix
    if curr_split > 0:
        curr_dir = curr_dir + '_split_%i' % (curr_split + 1)

    curr_meta_path = os.path.join(curr_dir, 'IT_ave', meta_name)
    cPickle.dump(IT_meta, open(curr_meta_path,'w'))

    curr_meta_path = os.path.join(curr_dir, 'V4_ave', meta_name)
    cPickle.dump(V4_meta, open(curr_meta_path,'w'))

    #break
