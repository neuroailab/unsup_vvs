import tensorflow as tf
import argparse
import os
import numpy as np
import sys

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_parser():
    parser = argparse.ArgumentParser(description='The script to write to tfrecords combining images and labels')
    parser.add_argument('--savedir', default = '/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label', type = str, action = 'store', help = 'Directory to save the tfrecords')
    parser.add_argument('--loaddir', default = '/mnt/fs1/Dataset/TFRecord_Imagenet_standard/', type = str, action = 'store', help = 'Directory to save the tfrecords')
    parser.add_argument('--staindx', default = 0, type = int, action = 'store', help = 'Relative index of the image to start')
    parser.add_argument('--lenindx', default = 1, type = int, action = 'store', help = 'Length of the tfrs to handle')
    parser.add_argument('--dataformat', default = 'train-%05i-of-01024-%s.tfrecords', type = str, action = 'store', help = 'Prefix for the tfrecords')

    return parser

def combine_one(image_file, label_file, output_file):
    image_iter = tf.python_io.tf_record_iterator(path=image_file)
    label_iter = tf.python_io.tf_record_iterator(path=label_file)

    writer = tf.python_io.TFRecordWriter(output_file)

    for image_record, label_record in zip(image_iter, label_iter):

        image_example = tf.train.Example()
        image_example.ParseFromString(image_record)

        img_string = image_example.features.feature['images'].bytes_list.value[0]

        label_example = tf.train.Example()
        label_example.ParseFromString(label_record)

        lbl_decode = int(label_example.features.feature['labels'].int64_list.value[0])

        example = tf.train.Example(features=tf.train.Features(feature={
            'images': _bytes_feature(img_string),
            'labels': _int64_feature(lbl_decode)}))
        writer.write(example.SerializeToString())

    writer.close()
    image_iter.close()
    label_iter.close()

def main():
    parser = get_parser()
    args = parser.parse_args()

    file_list = [( \
        os.path.join(args.loaddir, "images", args.dataformat % (tmp_indx, "images")), \
        os.path.join(args.loaddir, "labels_0", args.dataformat % (tmp_indx, "labels_0")), \
        os.path.join(args.savedir, args.dataformat % (tmp_indx, "image_label")) \
        ) for tmp_indx in xrange(args.staindx, args.staindx + args.lenindx)]
    file_list = filter(lambda x: os.path.exists(x[0]) and os.path.exists(x[1]), file_list)

    os.system('mkdir -p %s' % args.savedir)

    for image_file, label_file, output_file in file_list:
        combine_one(image_file, label_file, output_file)

if __name__=="__main__":
    main()
