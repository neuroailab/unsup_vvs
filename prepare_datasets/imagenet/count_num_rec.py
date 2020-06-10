import tensorflow as tf
import argparse
import os
import numpy as np
import sys
from tqdm import tqdm
import cPickle

def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to count number of records in tfrecords')
    parser.add_argument(
            '--save_path', 
            default='./instance_rec_count.pkl', 
            type=str, action='store', 
            help='Directory to save the results')
    parser.add_argument(
            '--load_dir', 
            default='/data2/chengxuz/Dataset/TFRecord_Imagenet_standard/image_label_full', 
            type=str, action='store', help='Directory to load the tfrecords')
    parser.add_argument(
            '--cut_num', 
            default=10, 
            type=int, action='store', 
            help='Number of cuts')
    return parser

def main():
    # Get arguments
    parser = get_parser()
    args = parser.parse_args()
    # Get tfrecords file list
    file_num = 1024
    dataformat = 'train-%05i-of-01024'
    file_list = [\
        os.path.join(args.load_dir, dataformat % (tmp_indx)) \
        for tmp_indx in xrange(0, file_num)]
    file_list = filter(lambda x: os.path.exists(x), file_list)
    file_list.sort()
    # Build sequences
    each_cut_len = int(file_num / args.cut_num)
    tfr_cut_seq = [
            (curr_cut*each_cut_len, (curr_cut+1) * each_cut_len)
            for curr_cut in range(args.cut_num)]
    tfr_cut_seq[-1] = (tfr_cut_seq[-1][0], 1024)
    # Do the counting
    res_list = []
    curr_start = 0
    curr_index = 0
    for curr_seq_idx, (curr_seq_sta,curr_seq_end) in enumerate(tfr_cut_seq):
        # Count in each tfr
        curr_start = curr_index
        res_list.append((curr_start, file_list[curr_seq_sta:curr_seq_end]))
        for input_file in tqdm(file_list[curr_seq_sta:curr_seq_end]):
            # Index offset, and which tensor to update
            input_iter = tf.python_io.tf_record_iterator(path=input_file)
            for curr_record in input_iter:
                curr_index += 1
    # Save the result
    cPickle.dump(res_list, open(args.save_path, 'w'))

if __name__=="__main__":
    main()
