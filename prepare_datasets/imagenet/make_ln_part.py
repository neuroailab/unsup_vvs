import os
import argparse
import sys
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='The script to random sample some tfrecords and build links in new folder')
    parser.add_argument('--savedir', default = '/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label_part', type = str, action = 'store', help = 'Directory to save the tfrecords')
    parser.add_argument('--loaddir', default = '/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label', type = str, action = 'store', help = 'Directory to load the tfrecords')
    parser.add_argument('--seed', default = 0, type = int, action = 'store', help = 'Random seed for sampling')
    parser.add_argument('--num', default = 1, type = int, action = 'store', help = 'Length of the tfrs to create links')
    parser.add_argument('--wholenum', default = 1024, type = int, action = 'store', help = 'Overall number of tfrs')
    parser.add_argument('--dataformat', default = 'train-%05i-of-01024-image_label.tfrecords', type = str, action = 'store', help = 'Prefix for the tfrecords')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    os.system('mkdir -p %s' % args.savedir)

    if args.num < args.wholenum:
        choose_nums = np.random.choice(args.wholenum, args.num, replace = False)
    else:
        choose_nums = range(args.wholenum)

    for each_num in choose_nums:
        curr_fname = args.dataformat % each_num
        source_file = os.path.join(args.loaddir, curr_fname)
        dest_file = os.path.join(args.savedir, curr_fname)

        os.system('ln -s %s %s' % (source_file, dest_file))
    os.system('ln -s %s %s' % (os.path.join(args.loaddir, 'validat*'), args.savedir))

if __name__=="__main__":
    main()
