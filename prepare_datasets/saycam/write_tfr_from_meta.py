import sys, os
import numpy as np
import argparse
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import pdb


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
            default='/mnt/fs4/chengxuz/Dataset/saycam_frames_tfr', type=str, 
            action='store', help='Directory to save the tfrecords')
    parser.add_argument(
            '--img_folder', 
            default='/data4/shetw/infant_headcam/jpgs_extracted', type=str, 
            action='store', help='Directory storing the original images')
    parser.add_argument(
            '--meta_file', 
            default='/data4/shetw/infant_headcam/metafiles/SAYCam_jpgs.txt', 
            type=str, action='store', help='Meta file listing the jpgs')
    parser.add_argument(
            '--random_seed', 
            default=0, type=int, 
            action='store', help='Random seed for numpy')
    return parser


def get_label_dict(folder):
    subfolders = os.listdir(folder)
    all_nouns = []
    for subfolder in subfolders:
        all_nouns.extend(os.listdir(os.path.join(folder, subfolder)))
    all_nouns.sort()
    label_dict = {noun:idx for idx, noun in enumerate(all_nouns)}
    return label_dict, all_nouns


def get_imgs_from_dir(synset_dir):
    curr_imgs = os.listdir(synset_dir)
    curr_imgs = [os.path.join(synset_dir, each_img) 
                 for each_img in curr_imgs]
    curr_imgs.sort()
    return curr_imgs


def get_path_and_label(img_folder, meta_path):
    all_path_labels = []
    label_dict, all_nouns = get_label_dict(img_folder)
    print('Getting all image paths')
    with open(meta_path, 'r') as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            subfolders = os.listdir(img_folder)
            curr_noun = line.split('/')[0]
            for subfolder in subfolders:
                if subfolder.startswith(curr_noun[0]):
                    break
            curr_path = os.path.join(
                    img_folder, subfolder, line.replace('\n', ''))
            all_path_labels.append((curr_path, label_dict[curr_noun]))
    return all_path_labels


def get_img_raw_str(jpg_path):
    with tf.gfile.FastGFile(jpg_path, 'rb') as f:
        img_raw_str = f.read()
    return img_raw_str


def write_to_tfrs(tfrs_path, curr_file_list):
    # Write each image and label
    writer = tf.python_io.TFRecordWriter(tfrs_path)
    for idx, jpg_path, lbl in curr_file_list:
        img_raw_str = get_img_raw_str(jpg_path)
        example = tf.train.Example(features=tf.train.Features(feature={
            'images': _bytes_feature(img_raw_str),
            'labels': _int64_feature(int(lbl)),
            'index': _int64_feature(int(idx)),
            }))
        writer.write(example.SerializeToString())
    writer.close()


def build_all_tfrs_from_folder(
        folder_path, meta_path, num_tfrs, tfr_pat, 
        random_seed=None):
    # get all path and labels, shuffle them if needed
    all_path_labels = get_path_and_label(folder_path, meta_path)
    if random_seed is not None:
        np.random.seed(random_seed)
    all_path_labels = np.random.permutation(all_path_labels)
    overall_num_imgs = len(all_path_labels)
    all_path_lbl_idx = [(idx, path, int(lbl)) \
                        for idx, (path, lbl) in enumerate(all_path_labels)]
    print('%i images in total' % overall_num_imgs)

    # Cut into num_tfr tfrecords and write each of them
    num_img_per = int(np.ceil(overall_num_imgs*1.0/num_tfrs))
    print('Writing into tfrecords')
    for curr_tfr in tqdm(range(num_tfrs)):
        tfrs_path = tfr_pat % (curr_tfr, num_tfrs)
        start_num = curr_tfr * num_img_per
        end_num = min((curr_tfr+1) * num_img_per, overall_num_imgs)
        write_to_tfrs(tfrs_path, all_path_lbl_idx[start_num:end_num])


def main():
    parser = get_parser()
    args = parser.parse_args()
    os.system('mkdir -p %s' % args.save_dir)

    build_all_tfrs_from_folder(
            args.img_folder, args.meta_file,
            1024,
            os.path.join(args.save_dir, 'train-%05i-of-%05i'), 
            args.random_seed)


if __name__=="__main__":
    main()
