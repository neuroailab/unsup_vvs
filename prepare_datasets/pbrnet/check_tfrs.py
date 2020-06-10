import os, sys
import numpy as np

import tensorflow as tf
import cPickle

import json
import copy
import argparse
from PIL import Image

from write_tfrs import get_args


#module load tensorflow/0.12.1
#module load anaconda/anaconda.4.2.0.python2.7

def read_it(folder_path, filetype, pngpattern):

    ret_list = []

    png_list = os.listdir(folder_path)
    png_list.sort()

    for png_name in png_list:

        if not pngpattern==None:
            if not pngpattern in png_name:
                continue

        img_path = os.path.join(folder_path, png_name)

        img = np.array(Image.open(img_path))

        if filetype=='uint16':
            img = img.astype(np.uint16)

        if len(img.shape)==2:
            img = img[:, :, np.newaxis]

        ret_list.append(img)

    return ret_list

def load_it(tfrecords_filename, filetype, keyname, sess):
    reconstructed_images = []

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    get_num = 0

    for string_record in record_iterator:
        
        example = tf.train.Example()
        example.ParseFromString(string_record)
        
        img_string = (example.features.feature[keyname]
                                      .bytes_list
                                      .value[0])

        if filetype=='uint8':
            img_vector = tf.image.decode_png(img_string)
        else:
            img_vector = tf.image.decode_png(img_string, dtype = tf.uint16)

        img_array = sess.run(img_vector)

        reconstructed_images.append(img_array)

    return reconstructed_images


def main():

    args = get_args()

    all_folders = os.listdir(args.loaddir)
    all_folders.sort()

    sta_indx = args.indx * args.includelen
    end_indx = min((args.indx+1) * args.includelen, len(all_folders))

    tfr_path = os.path.join(args.savedir, "%s%i.tfrecords" % (args.tfrprefix, args.indx))

    sess = tf.Session()

    recon_array = load_it(tfr_path, args.filetype, args.keyname, sess)

    print(len(recon_array))

    all_orig_array = []

    for folder_name in all_folders[sta_indx:end_indx]:
        all_orig_array.extend(read_it(os.path.join(args.loaddir, folder_name), args.filetype, args.pngpattern))

    print(len(all_orig_array))

    for recon_img, orig_img in zip(recon_array, all_orig_array):
        print(np.allclose(recon_img, orig_img))
        print(np.max(orig_img))

if __name__=='__main__':
    main()
