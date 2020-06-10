import sys, os
import numpy as np
from tqdm import tqdm

import argparse

def get_parser():
    parser = argparse.ArgumentParser(
            description='Sample some images from train image folder (for pytorch implementations)')
    parser.add_argument(
            '--save_dir', 
            default='/data5/chengxuz/imagenet_raw/val_train', type=str, 
            action='store', help='Directory to make the validation folder from train images')
    parser.add_argument(
            '--img_folder', 
            default='/data5/chengxuz/imagenet_raw/train', type=str, 
            action='store', help='Directory storing the original images')
    parser.add_argument(
            '--img_per_dir', 
            default=50, type=int, 
            action='store', help='Number of images per directory')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Set the random seed
    np.random.seed(0)

    # Get folders
    curr_dir_list = os.listdir(args.img_folder)

    # Sample images from each folder
    os.system('mkdir -p %s' % args.save_dir)
    for each_dir in tqdm(curr_dir_list):
        prev_dir = os.path.join(args.img_folder, each_dir)
        new_dir = os.path.join(args.save_dir, each_dir)
        os.system('mkdir -p %s' % new_dir)

        ## Get img list
        curr_img_list = os.listdir(prev_dir)
        assert len(curr_img_list) > args.img_per_dir, "Number of images per dir should be smaller than %i" % len(curr_img_list)
        sample_img_list = np.random.choice(curr_img_list, args.img_per_dir, replace=False)

        ## make symbolic links in each new folder
        for each_img in sample_img_list:
            new_img_path = os.path.join(new_dir, each_img)
            prev_img_path = os.path.join(prev_dir, each_img)

            os.system('ln -s %s %s' % (prev_img_path, new_img_path))

if __name__=="__main__":
    main()

