from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import argparse

def main():
    parser = argparse.ArgumentParser(description='The script to move some tfrecords')
    parser.add_argument('--fromdir', default = '/mnt/fs0/chengxuz/Data/pbrnet_folder/tfrecords', type = str, action = 'store', help = 'Path to the folder containing tfrecords')
    parser.add_argument('--todir', default = '/mnt/fs0/chengxuz/Data/pbrnet_folder/tfrecords_val', type = str, action = 'store', help = 'Path to the folder saving tfrecords')
    parser.add_argument('--randseed', default = 0, type = int, action = 'store', help = 'Seed for random')
    parser.add_argument('--testlen', default = 50, type = int, action = 'store', help = 'Length of test')

    args    = parser.parse_args()

    np.random.seed(args.randseed)
    indxlist = np.random.choice(os.listdir(os.path.join(args.fromdir, 'mlt')), args.testlen, replace = False)

    for curr_key in ['category', 'depth', 'mlt', 'normal', 'valid']:
        target_dir = os.path.join(args.todir, curr_key)
        os.system('mkdir -p %s' % target_dir)

        for curr_name in indxlist:
            from_tfr = os.path.join(args.fromdir, curr_key, curr_name)
            to_tfr = os.path.join(args.todir, curr_key, curr_name)
            if os.path.exists(from_tfr):
                cmd_str = 'mv %s %s' % (from_tfr, to_tfr)
                print(cmd_str)
                #os.system(cmd_str)

if __name__ == '__main__':
    main()
