import sys, os
import numpy as np
import argparse
import tensorflow as tf
from PIL import Image
from build_new_imagenet_raw import get_imgs_from_dir
from tqdm import tqdm
from balance_sample import blacklist, write_jpg_lbls_to_tfrs

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_parser():
    parser = argparse.ArgumentParser(
            description='Generate tfrecords from jpeg images, ' \
                    + 'default parameters are for node7')
    parser.add_argument(
            '--save_dir', 
            default='/mnt/fs1/Dataset/new_imagenet_tfr', type=str, 
            action='store', help='Directory to save the tfrecords')
    parser.add_argument(
            '--img_folder', 
            default='/data5/chengxuz/new_imagenet_raw/train', type=str, 
            action='store', help='Directory storing the original images')
    parser.add_argument(
            '--num_tfr', 
            default=512, type=int, 
            action='store', help='Number of overall tfrecords')
    parser.add_argument(
            '--random_seed', 
            default=0, type=int, 
            action='store', help='Random seed for numpy')
    parser.add_argument(
            '--tfr_name_pat', 
            default='train-%03i-of-%03i', type=str, 
            action='store', help='Pattern of tfrecord names')
    return parser

def get_label_dict(args):
    all_nouns = os.listdir(args.img_folder)
    all_nouns.sort()
    label_dict = {noun:idx for idx, noun in enumerate(all_nouns)}
    return label_dict, all_nouns

def get_path_and_label(args):
    all_path_labels = []
    label_dict, all_nouns = get_label_dict(args)
    print('Getting all image paths')
    for each_noun in tqdm(all_nouns):
        curr_paths = get_imgs_from_dir(
                os.path.join(args.img_folder, each_noun))
        curr_paths = filter(
                lambda x: np.sum(
                    [each_black in x for each_black in blacklist]
                    )==0, 
                curr_paths)
        curr_path_labels = [(each_path, label_dict[each_noun]) \
                for each_path in curr_paths]
        all_path_labels.extend(curr_path_labels)
    return all_path_labels

def main():
    parser = get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.system('mkdir -p %s' % args.save_dir)
    # get all path and labels, shuffle them
    np.random.seed(args.random_seed)
    all_path_labels = get_path_and_label(args)
    all_path_labels = np.random.permutation(all_path_labels)
    overall_num_imgs = len(all_path_labels)
    print('%i images in total' % overall_num_imgs)
    # Cut into num_tfr tfrecords and write each of them
    num_img_per = int(np.ceil(overall_num_imgs*1.0/args.num_tfr))
    print('Writing into tfrecords')
    for curr_tfr in tqdm(range(args.num_tfr)):
        tfrs_path = os.path.join(args.save_dir, 
                                 args.tfr_name_pat % (curr_tfr, args.num_tfr))
        start_num = curr_tfr * num_img_per
        end_num = min((curr_tfr+1) * num_img_per, overall_num_imgs)
        write_jpg_lbls_to_tfrs(tfrs_path, all_path_labels[start_num:end_num])

if __name__=="__main__":
    main()
