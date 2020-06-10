import os, sys
import numpy as np

import tensorflow as tf
import cPickle

import json
import copy
import argparse
from PIL import Image


#module load tensorflow/0.12.1
#module load anaconda/anaconda.4.2.0.python2.7

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_it(writer, folder_path, filetype, sess, keyname, img_place, img_raw, pngpattern):
    png_list = os.listdir(folder_path)
    png_list.sort()

    for png_name in png_list:

        if not pngpattern==None:
            if not pngpattern in png_name:
                continue

        img_path = os.path.join(folder_path, png_name)

        img = np.array(Image.open(img_path))

        if len(img.shape)==2:
            img = img[:, :, np.newaxis]

        if filetype=='uint16':
            img = img.astype(np.uint16)
            #print(img[0,0,0])
            #img = tf.constant(img, dtype = tf.uint16)

        img_raw_str = sess.run(img_raw, feed_dict = {img_place: img})

        example = tf.train.Example(features=tf.train.Features(feature={
            keyname: _bytes_feature(img_raw_str)}))
        writer.write(example.SerializeToString())

def get_args():
    # Arguments parsing
    parser = argparse.ArgumentParser(description='The script to train the combine net')
    parser.add_argument('--loaddir', default = '/scratch/users/chengxuz/Data/pbrnet/', type = str, action = 'store', help = 'Load directory of the original PBRnet')
    parser.add_argument('--savedir', default = '/scratch/users/chengxuz/Data/pbrnet/tfrecords/', type = str, action = 'store', help = 'Save directory')
    parser.add_argument('--includelen', default = 2, type = int, action = 'store', help = 'Number of folders to include')
    parser.add_argument('--indx', default = 0, type = int, action = 'store', help = 'Index of tfrs')
    parser.add_argument('--indxlen', default = 1, type = int, action = 'store', help = 'Index of tfrs')
    parser.add_argument('--filetype', default = 'uint8', type = str, action = 'store', help = 'Type of files to load')
    parser.add_argument('--tfrprefix', default = 'pbrnet_', type = str, action = 'store', help = 'Prefix of the tfrecords')
    parser.add_argument('--keyname', default = 'mlt', type = str, action = 'store', help = 'Name for the key to save in tfrcs')
    parser.add_argument('--pngpattern', default = None, type = str, action = 'store', help = 'PNG file name pattern to load')
    
    args = parser.parse_args()

    if args.pngpattern==None:
        args.loaddir = os.path.join(args.loaddir, args.keyname)
        args.savedir = os.path.join(args.savedir, args.keyname)

    return args

def main():

    args = get_args()
    
    os.system('mkdir -p %s' % args.savedir)

    all_folders = os.listdir(args.loaddir)
    all_folders.sort()
    print(len(all_folders))

    if args.filetype=='uint8':
        img_place = tf.placeholder(dtype=tf.uint8)
    else:
        img_place = tf.placeholder(dtype=tf.uint16)

    img_raw = tf.image.encode_png(img_place)

    sess = tf.Session()

    for curr_indx in xrange(args.indx, args.indxlen + args.indx):

        sta_indx = curr_indx * args.includelen
        end_indx = min((curr_indx+1) * args.includelen, len(all_folders))

        if end_indx <= sta_indx:
            break

        tfr_path = os.path.join(args.savedir, "%s%i.tfrecords" % (args.tfrprefix, curr_indx))
        writer = tf.python_io.TFRecordWriter(tfr_path)

        for folder_name in all_folders[sta_indx:end_indx]:
            write_it(writer, os.path.join(args.loaddir, folder_name), args.filetype, sess, args.keyname, img_place, img_raw, args.pngpattern)

        writer.close()

if __name__=='__main__':
    main()
