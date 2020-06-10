import tensorflow as tf
import argparse
import os
import numpy as np
import sys
import h5py

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        
def write_one(args, data_fin, label_fin, curr_stahdf5indx, curr_endhdf5indx, curr_indx, withindx = 0):

    all_images = np.asarray(data_fin['data'][curr_stahdf5indx : curr_endhdf5indx])
    all_labels = np.asarray(label_fin['labels'][curr_stahdf5indx : curr_endhdf5indx])

    tfr_filename = os.path.join(args.savedir, args.dataformat % curr_indx)
    writer = tf.python_io.TFRecordWriter(tfr_filename)

    for curr_hdf5indx in xrange(curr_endhdf5indx - curr_stahdf5indx):
        now_image = all_images[curr_hdf5indx]
        now_image = now_image.reshape([3, 256, 256])
        now_image = np.transpose(now_image, [1,2,0])
        
        image_raw = now_image.tostring()

        if withindx==0:
            example = tf.train.Example(features=tf.train.Features(feature={
                'images': _bytes_feature(image_raw),
                'labels': _int64_feature(all_labels[curr_hdf5indx])}))
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'images': _bytes_feature(image_raw),
                'labels': _int64_feature(all_labels[curr_hdf5indx]),
                'index': _int64_feature(curr_hdf5indx + curr_stahdf5indx),
                }))
        writer.write(example.SerializeToString())

    writer.close()


def get_parser():
    parser = argparse.ArgumentParser(description='The script to write to tfrecords combining images and labels')
    parser.add_argument('--savedir', default = '/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label_hdf5', type = str, action = 'store', help = 'Directory to save the tfrecords')
    parser.add_argument('--loaddata', default = '/mnt/fs1/Dataset/imagenet_hdf5/data.raw', type = str, action = 'store', help = 'Directory to save the tfrecords')
    parser.add_argument('--loadlabel', default = '/mnt/fs1/Dataset/imagenet_hdf5/labels.hdf5', type = str, action = 'store', help = 'Directory to save the tfrecords')
    parser.add_argument('--staindx', default = 0, type = int, action = 'store', help = 'Relative index of the tfreord to start')
    parser.add_argument('--lenindx', default = 1, type = int, action = 'store', help = 'Length of the tfrs to handle')
    parser.add_argument('--stahdf5indx', default = 0, type = int, action = 'store', help = 'hdf5 indx to start')
    parser.add_argument('--endhdf5indx', default = 1240129, type = int, action = 'store', help = 'hdf5 indx for ending')
    parser.add_argument('--leneach', default = 2481, type = int, action = 'store', help = 'number of records in each tfrecord file')
    parser.add_argument('--dataformat', default = 'train-%05i-of-00500-image_label.tfrecords', type = str, action = 'store', help = 'Prefix for the tfrecords')
    parser.add_argument('--withindx', default = 0, type = int, action = 'store', help = 'Whether storing the index at the same time')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    data_fin = h5py.File(args.loaddata, 'r')
    label_fin = h5py.File(args.loadlabel, 'r')

    os.system('mkdir -p %s' % args.savedir)

    for curr_indx in xrange(args.staindx, args.staindx + args.lenindx):
        curr_stahdf5indx = args.stahdf5indx + curr_indx * args.leneach
        curr_endhdf5indx = min(curr_stahdf5indx + args.leneach, args.endhdf5indx)

        if curr_endhdf5indx<=curr_stahdf5indx:
            break
        
        write_one(args, data_fin, label_fin, curr_stahdf5indx, curr_endhdf5indx, curr_indx, args.withindx)

    data_fin.close()
    label_fin.close()

if __name__=="__main__":
    main()
