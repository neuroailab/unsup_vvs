import os, sys
import numpy as np

import tensorflow as tf
import cPickle

import json
import copy
import argparse
from PIL import Image


# module load tensorflow/0.12.1
# module load anaconda/anaconda.4.2.0.python2.7

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_it(writer, folder_path, filetype, sess, keyname, img_place, img_raw, pngpattern):
    png_list = os.listdir(folder_path)
    png_list.sort()

    for png_name in png_list:

        if not pngpattern == None:
            if not pngpattern in png_name:
                continue

        img_path = os.path.join(folder_path, png_name)

        img = np.array(Image.open(img_path))

        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]

        if filetype == 'uint16':
            img = img.astype(np.uint16)
            # print(img[0,0,0])
            # img = tf.constant(img, dtype = tf.uint16)

        img_raw_str = sess.run(img_raw, feed_dict={img_place: img})

        if keyname.find(',') != -1:
            keyname1 = keyname.split(',')[0]
            keyname2 = keyname.split(',')[1]
            example = tf.train.Example(features=tf.train.Features(feature={
                keyname1: _bytes_feature(img_raw_str),
                keyname2: _bytes_feature(img_raw_str_2)
            }))
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                keyname: _bytes_feature(img_raw_str)}))
        writer.write(example.SerializeToString())


def get_args():
    # Arguments parsing
    parser = argparse.ArgumentParser(description='The script to train the combine net')
    parser.add_argument('--loaddir', default='/mnt/fs1/Dataset/scenenet_all/scenenet_new_val/', type=str, action='store',
                        help='Load directory of the original PBRnet')
    parser.add_argument('--savedir', default='/mnt/fs1/siming/Dataset/scenenet/tfrecords/', type=str, action='store',
                        help='Save directory')
    parser.add_argument('--includelen', default=2, type=int, action='store', help='Number of folders to include')
    parser.add_argument('--indx', default=0, type=int, action='store', help='Index of tfrs')
    parser.add_argument('--indxlen', default=1, type=int, action='store', help='Index of tfrs')
    parser.add_argument('--filetype', default='uint8', type=str, action='store', help='Type of files to load')
    parser.add_argument('--tfrprefix', default='validation-', type=str, action='store', help='Prefix of the tfrecords')
    parser.add_argument('--keyname', default='mlt,depth', type=str, action='store', help='Name for the key to save in tfrcs')
    parser.add_argument('--pngpattern', default=None, type=str, action='store', help='PNG file name pattern to load')

    args = parser.parse_args()
    '''
    if args.pngpattern == None:
        args.loaddir = os.path.join(args.loaddir, args.keyname)
        args.savedir = os.path.join(args.savedir, args.keyname)
        '''

    return args


def main():
    args = get_args()

    os.system('mkdir -p %s' % args.savedir)

    depth_folder = args.loaddir + 'depth/'
    normal_folder = args.loaddir + 'normal/'
    mlt_folder = args.loaddir + 'photo/'

    filenames = os.listdir(depth_folder)
    print(len(filenames))
    curr_index = 0
    for filename in filenames:
        print(curr_index)
        print(filename)
        if filename == 'meta.pkl':
            continue
        reconstructed_mlts = []
        mlt_record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(mlt_folder, filename))
        for string_record in mlt_record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            reconstructed_mlt = (example.features.feature['photo'].bytes_list.value[0])

            reconstructed_mlts.append(reconstructed_mlt)

        reconstructed_depths = []
        depth_record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(depth_folder, filename))
        for string_record in depth_record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            reconstructed_depth = (example.features.feature['depth'].bytes_list.value[0])

            reconstructed_depths.append(reconstructed_depth)

        print(len(reconstructed_depths))
        print(len(reconstructed_mlts))

        #mlt_place = tf.placeholder(dtype=tf.uint8)
        #mlt_raw = tf.image.encode_png(mlt_place)

        #sess1 = tf.Session()

        #depth_place = tf.placeholder(dtype=tf.uint16)
        #depth_raw = tf.image.encode_png(mlt_place)

        #sess2 = tf.Session()

        tfr_path = os.path.join(args.savedir, "%s%i.tfrecords" % (args.tfrprefix, curr_index))
        print(tfr_path)
        writer = tf.python_io.TFRecordWriter(tfr_path)

        for i in range(0, len(reconstructed_depths)):

            #depth_img_raw_str = sess2.run(depth_raw, feed_dict={depth_place: reconstructed_depths[i]})
            #mlt_img_raw_str = sess1.run(mlt_raw, feed_dict={mlt_place: reconstructed_mlts[i]})

            example = tf.train.Example(features=tf.train.Features(feature={
                'mlt': _bytes_feature(reconstructed_mlts[i]),
                'depth': _bytes_feature(reconstructed_depths[i])}))

            writer.write(example.SerializeToString())

        writer.close()

        curr_index = curr_index + 1






    '''
    if args.filetype == 'uint8':
        img_place = tf.placeholder(dtype=tf.uint8)
    else:
        img_place = tf.placeholder(dtype=tf.uint16)

    img_raw = tf.image.encode_png(img_place)

    sess = tf.Session()

    for curr_indx in xrange(args.indx, args.indxlen + args.indx):

        sta_indx = curr_indx * args.includelen
        end_indx = min((curr_indx + 1) * args.includelen, len(all_folders))

        if end_indx <= sta_indx:
            break

        tfr_path = os.path.join(args.savedir, "%s%i.tfrecords" % (args.tfrprefix, curr_indx))
        writer = tf.python_io.TFRecordWriter(tfr_path)

        for folder_name in all_folders[sta_indx:end_indx]:
            write_it(writer, os.path.join(args.loaddir, folder_name), args.filetype, sess, args.keyname, img_place,
                     img_raw, args.pngpattern)

        writer.close()
    '''


if __name__ == '__main__':
    main()
