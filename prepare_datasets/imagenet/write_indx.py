import tensorflow as tf
import argparse
import os
import numpy as np
import sys
from tqdm import tqdm

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_parser():
    parser = argparse.ArgumentParser(description='The script to write indexes to tfrecords with images and labels')
    parser.add_argument('--savedir', default = '/data2/chengxuz/Dataset/TFRecord_Imagenet_standard/image_label_full_widx', 
            type = str, action = 'store', help = 'Directory to save the tfrecords')
    parser.add_argument('--loaddir', default = '/data2/chengxuz/Dataset/TFRecord_Imagenet_standard/image_label_full', 
            type = str, action = 'store', help = 'Directory to load the tfrecords')
    parser.add_argument('--staindx', default = 0, type = int, action = 'store', 
            help = 'Relative index of the image to start')
    parser.add_argument('--lenindx', default = 1, type = int, action = 'store', 
            help = 'Length of the tfrs to handle')
    parser.add_argument('--dataformat', default = 'train-%05i-of-01024', type = str, action = 'store', 
            help = 'Prefix for the tfrecords')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    file_list = [( \
        os.path.join(args.loaddir, args.dataformat % (tmp_indx)), \
        os.path.join(args.savedir, args.dataformat % (tmp_indx)) \
        ) for tmp_indx in xrange(args.staindx, args.staindx + args.lenindx)]
    file_list = filter(lambda x: os.path.exists(x[0]), file_list)

    os.system('mkdir -p %s' % args.savedir)

    per_tfr_num = 1251
    idx_now = per_tfr_num*args.staindx
    for input_file, output_file in tqdm(file_list):
        input_iter = tf.python_io.tf_record_iterator(path=input_file)
        output_iter = tf.python_io.TFRecordWriter(output_file)

        for curr_record in input_iter:
            image_example = tf.train.Example()
            image_example.ParseFromString(curr_record)

            img_string = image_example.features.feature['images'].bytes_list.value[0]
            lbl_decode = int(image_example.features.feature['labels'].int64_list.value[0])

            example = tf.train.Example(features=tf.train.Features(feature={
                'images': _bytes_feature(img_string),
                'labels': _int64_feature(lbl_decode),
                'index': _int64_feature(idx_now),
                }))
            output_iter.write(example.SerializeToString())
            idx_now += 1
        input_iter.close()
        output_iter.close()
    print(idx_now)
    os.system('ln -s %s %s' % (os.path.join(args.loaddir, 'validat*'), args.savedir))

if __name__=="__main__":
    main()
