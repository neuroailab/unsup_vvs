import tensorflow as tf
import argparse
import os
import numpy as np
import sys
from tqdm import tqdm
import pickle

def get_parser():
    parser = argparse.ArgumentParser(
            description='Collect labels from tfrecords ' \
                    + 'with images, labels, and indexs')
    parser.add_argument(
            '--save_path', 
            default=None, type=str, required=True,
            action='store', help='Directory to save the pkl file')
    parser.add_argument(
            '--load_dir', 
            default=None, type=str, required=True,
            action='store', help='Directory to load the tfrecords')
    parser.add_argument(
            '--data_format', default='train-%05i-of-01024', 
            type=str, action='store', 
            help='Data format for the tfrecords')
    parser.add_argument(
            '--num_tfr', default=1024, 
            type=int, action='store', 
            help='Number of the tfrecords')
    parser.add_argument(
            '--num_img', default=1281167,
            type=int, action='store', 
            help='Number of images')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    # Get file list
    file_list = [\
        os.path.join(args.load_dir, args.data_format % (tmp_indx)) \
        for tmp_indx in range(args.num_tfr)]
    file_list = filter(lambda x: os.path.exists(x), file_list)
    # Get label array
    label_arr = np.zeros(args.num_img)
    for input_file in tqdm(file_list):
        input_iter = tf.python_io.tf_record_iterator(path=input_file)
        for curr_record in input_iter:
            image_example = tf.train.Example()
            image_example.ParseFromString(curr_record)
            ## Get label and index for each record
            label_decode = int(image_example.features.feature['labels']\
                    .int64_list.value[0])
            index_decode = int(image_example.features.feature['index']\
                    .int64_list.value[0])
            label_arr[index_decode] = label_decode
        input_iter.close()
    # Write to pickle
    pickle.dump(label_arr, open(args.save_path, 'wb'))

if __name__=="__main__":
    main()
