import tensorflow as tf
import argparse
import os
import numpy as np
import sys
import pdb

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_parser():
    parser = argparse.ArgumentParser(description='The script to write to tfrecords combining images and labels')
    parser.add_argument('--savedir', default = '/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label_full', type = str, action = 'store', help = 'Directory to save the tfrecords')
    parser.add_argument('--loaddir', default = '/mnt/fs1/Dataset/TFRecord_Imagenet_standard/full', type = str, action = 'store', help = 'Directory to save the tfrecords')
    parser.add_argument('--staindx', default = 0, type = int, action = 'store', help = 'Relative index of the image to start')
    parser.add_argument('--lenindx', default = 1, type = int, action = 'store', help = 'Length of the tfrs to handle')
    parser.add_argument('--dataformat', default = 'train-%05i-of-01024', type = str, action = 'store', help = 'Prefix for the tfrecords')

    return parser

def split_one(output_file, input_file):
    full_iter = tf.python_io.tf_record_iterator(path=input_file)

    writer = tf.python_io.TFRecordWriter(output_file)

    for tmp_record in full_iter:

        image_example = tf.train.Example()
        image_example.ParseFromString(tmp_record)
        img_string = image_example.features.feature['image/encoded'].bytes_list.value[0]
        lbl_decode = int(image_example.features.feature['image/class/label'].int64_list.value[0])

        example = tf.train.Example(features=tf.train.Features(feature={
            'images': _bytes_feature(img_string),
            'labels': _int64_feature(lbl_decode-1)}))
        writer.write(example.SerializeToString())

    full_iter.close()
    writer.close()

def main():
    parser = get_parser()
    args = parser.parse_args()

    file_list = [( \
        os.path.join(args.savedir, args.dataformat % tmp_indx), \
        os.path.join(args.loaddir, args.dataformat % tmp_indx) \
        ) for tmp_indx in xrange(args.staindx, args.staindx + args.lenindx)]
    file_list = filter(lambda x: os.path.exists(x[1]), file_list)

    os.system('mkdir -p %s' % args.savedir)

    for output_file, input_file in file_list:
        split_one(output_file, input_file)

if __name__=="__main__":
    main()
