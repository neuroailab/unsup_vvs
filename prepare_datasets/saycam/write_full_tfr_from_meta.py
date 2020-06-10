import sys, os
import numpy as np
import argparse
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import pdb
import write_tfr_from_meta


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
            default='/mnt/fs4/chengxuz/Dataset/saycam_all_frames_tfr', type=str, 
            action='store', help='Directory to save the tfrecords')
    parser.add_argument(
            '--img_folder', 
            default='/data4/shetw/infant_headcam/jpgs_extracted', type=str, 
            action='store', help='Directory storing the original images')
    parser.add_argument(
            '--meta_file', 
            default='/mnt/fs3/shetw/infant_headcam/metafiles/infant_10s_clips_metafile_v2.txt', 
            type=str, action='store', help='Meta file listing the jpgs')
    parser.add_argument(
            '--random_seed', 
            default=0, type=int, 
            action='store', help='Random seed for numpy')
    return parser


def get_path_and_label(img_folder, meta_path):
    all_path_lbl_idx = []
    label_dict, _ = write_tfr_from_meta.get_label_dict(img_folder)
    with open(meta_path, 'r') as fin:
        lines = fin.readlines()
        curr_idx = 0
        for line in tqdm(lines):
            _img_path, _sta_idx, _, _ = line.split(' ')
            curr_noun = _img_path.split('/')[1]
            curr_path = os.path.join(
                    img_folder, _img_path, '%06i.jpg')
            _sta_idx = int(_sta_idx)
            for _idx in range(125):
                all_path_lbl_idx.append(
                        (curr_idx, curr_path % (_sta_idx + _idx + 1), 
                         int(label_dict[curr_noun])))
                curr_idx += 1
    return all_path_lbl_idx


def build_all_tfrs_from_folder(
        folder_path, meta_path, num_tfrs, tfr_pat, 
        random_seed=None):
    # get all path and labels, shuffle them if needed
    all_path_lbl_idx = get_path_and_label(folder_path, meta_path)
    if random_seed is not None:
        np.random.seed(random_seed)
    all_path_lbl_idx = np.random.permutation(all_path_lbl_idx)
    overall_num_imgs = len(all_path_lbl_idx)
    print('%i images in total' % overall_num_imgs)

    # Cut into num_tfr tfrecords and write each of them
    num_img_per = int(np.ceil(overall_num_imgs*1.0/num_tfrs))
    print('Writing into tfrecords')
    for curr_tfr in tqdm(range(num_tfrs)):
        tfrs_path = tfr_pat % (curr_tfr, num_tfrs)
        start_num = curr_tfr * num_img_per
        end_num = min((curr_tfr+1) * num_img_per, overall_num_imgs)
        write_tfr_from_meta.write_to_tfrs(tfrs_path, all_path_lbl_idx[start_num:end_num])


def main():
    parser = get_parser()
    args = parser.parse_args()
    os.system('mkdir -p %s' % args.save_dir)

    build_all_tfrs_from_folder(
            args.img_folder, args.meta_file,
            10240,
            os.path.join(args.save_dir, 'train-%05i-of-%05i'), 
            args.random_seed)


if __name__=="__main__":
    main()
