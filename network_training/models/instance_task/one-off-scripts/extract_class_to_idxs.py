# Script for extracting a mapping from class to list of indices
import tensorflow as tf
import os
import numpy as np
import sys

import json
from collections import defaultdict

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def main():
    loaddir = '/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label_full_widx'
    dataformat = 'train-%05i-of-01024'
    file_list = [os.path.join(loaddir, dataformat % (tmp_idx,)) for tmp_idx in xrange(1024)]
    classes = defaultdict(list)
    
    for file_idx, record_file in enumerate(file_list):
        input_iter = tf.python_io.tf_record_iterator(path=record_file)

        nrecords = 0
        for curr_record in input_iter:
            image_example = tf.train.Example()
            image_example.ParseFromString(curr_record)

            #img_string = image_example.features.feature['images'].bytes_list.value[0]
            lbl_decode = int(image_example.features.feature['labels'].int64_list.value[0])
            idx_decode = int(image_example.features.feature['index'].int64_list.value[0])

            classes[lbl_decode].append(idx_decode)
            nrecords += 1

        print 'Processed', nrecords, 'records from file', record_file
            
    for k, v in classes.iteritems():
        print "class", k, "has", len(v), "examples"
    with open('./class_to_idxs_mapping.json', 'w') as f:
        f.write(json.dumps(classes))
    
if __name__=="__main__":
    main()
                                                                                                                                                                
