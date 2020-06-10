import sys, os
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import pdb

import argparse
import tensorflow as tf
from PIL import Image

blacklist = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
           'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
           'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
           'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
           'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
           'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
           'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
           'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
           'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
           'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
           'n07583066_647.JPEG', 'n13037406_4650.JPEG', 
           'n02105855_2933.JPEG', 'ILSVRC2012_val_00019877.JPEG']

def load_dict(meta_path, num_cat = 1000):
    data = sio.loadmat(meta_path)
    label_dict = {}
    label_list = []
    for indx_cat in xrange(num_cat):
        curr_synset = data['synsets'][indx_cat][0][1][0]
        curr_label = int(data['synsets'][indx_cat][0][0][0][0]) - 1
        label_dict[curr_synset] = curr_label
        label_list.append(curr_synset)

    return label_dict, label_list

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_parser():
    parser = argparse.ArgumentParser(
            description='Generate tfrecords from original images ' \
                    + 'for balanced random sampling, ' \
                    + 'default parameters are for node7')
    parser.add_argument(
            '--save_dir', 
            default='/mnt/fs1/Dataset/imagenet_p01', type=str, 
            action='store', help='Directory to save the tfrecords')
    parser.add_argument(
            '--rand_per', 
            default=0.01, type=float, 
            action='store', help='Percent for balanced sampling')
    parser.add_argument(
            '--img_folder', 
            default='/data5/chengxuz/Dataset/imagenet_raw/train', type=str, 
            action='store', help='Directory storing the original images')
    parser.add_argument(
            '--meta_path', 
            default='/home/chengxuz/ILSVRC2012_devkit_t12/data/meta.mat', 
            type=str, action='store', help='Path to the meta file')
    parser.add_argument(
            '--num_tfr', 
            default=512, type=int, 
            action='store', help='Number of images per tfrecord')
    parser.add_argument(
            '--tfr_name_pat', 
            default='train-%05i-of-%05i', type=str, 
            action='store', help='Pattern of tfrecord names')
    return parser

def write_jpg_lbls_to_tfrs(tfrs_path, curr_file_list):
    # Write each image and label
    writer = tf.python_io.TFRecordWriter(tfrs_path)
    for jpg_path, lbl in curr_file_list:
        with tf.gfile.FastGFile(jpg_path, 'rb') as f:
            img_raw_str = f.read()
        example = tf.train.Example(features=tf.train.Features(feature={
            'images': _bytes_feature(img_raw_str),
            'labels': _int64_feature(int(lbl))
            }))
        writer.write(example.SerializeToString())
    writer.close()

def main():
    parser = get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.system('mkdir -p %s' % args.save_dir)
    # Load the meta: correspondence between synset id and 0 to 1000 label
    label_dict, label_list = load_dict(args.meta_path)
    # Set the random seed
    np.random.seed(0)
    # Get the sampled files
    all_file_list = []
    label_list = os.listdir(args.img_folder)
    for curr_label in label_list:
        curr_dir = os.path.join(args.img_folder, curr_label)
        curr_file_list = os.listdir(curr_dir)
        curr_file_list.sort()
        # Get sampled files for this label
        curr_file_list = np.random.permutation(curr_file_list)
        curr_len = int(np.round(len(curr_file_list) * args.rand_per))
        curr_file_list = curr_file_list[:curr_len]
        # Filter out images in blacklist 
        curr_file_list = filter(
                lambda x: np.sum(
                    [each_black in x for each_black in blacklist]
                    )==0, 
                curr_file_list)
        # Append the file and label
        for curr_file in curr_file_list:
            all_file_list.append((
                os.path.join(args.img_folder, curr_label, curr_file), 
                label_dict[curr_label]))
    all_file_list = np.random.permutation(all_file_list)
    # Write tfrs
    overall_num_imgs = len(all_file_list)
    img_num_per_tfr = int(np.ceil(overall_num_imgs*1.0/args.num_tfr))
    tfr_name_pat = args.tfr_name_pat
    # Write each tfrecords
    for curr_tfr_indx in tqdm(range(0, args.num_tfr)):
        sta_num = curr_tfr_indx * img_num_per_tfr
        end_num = min(sta_num + img_num_per_tfr, overall_num_imgs)
        tfrs_name = tfr_name_pat % (curr_tfr_indx, args.num_tfr)
        tfrs_path = os.path.join(args.save_dir, tfrs_name)
        write_jpg_lbls_to_tfrs(tfrs_path, all_file_list[sta_num:end_num])

if __name__=="__main__":
    main()
