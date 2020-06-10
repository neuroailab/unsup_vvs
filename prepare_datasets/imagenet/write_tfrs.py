import tensorflow as tf
import argparse
import os
import numpy as np
import sys
from PIL import Image
import subprocess

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_parser():
    parser = argparse.ArgumentParser(description='The script to write to tfrecords from original images using tensorflow, for places')
    parser.add_argument('--savedir', default = '/scratch/users/chengxuz/Data/imagenet_tfr/tfrs_train', type = str, action = 'store', help = 'Directory to save the tfrecords')
    parser.add_argument('--staindx', default = 0, type = int, action = 'store', help = 'Relative index of the image to start')
    parser.add_argument('--lenindx', default = 1, type = int, action = 'store', help = 'Length of the tfrs to handle')
    parser.add_argument('--sshfolder', default = '/mnt/fs1/Dataset/imagenet_again/tfr_train', type = str, action = 'store', help = 'Folder on neuroaicluster to save to')
    parser.add_argument('--sshprefix', default = 'chengxuz@node9-neuroaicluster', type = str, action = 'store', help = 'which node to transfer to')
    parser.add_argument('--txtprefix', default = '/scratch/users/chengxuz/Data/imagenet_devkit/fname_', type = str, action = 'store', help = 'Txt file containing all the jpg files and labels')
    parser.add_argument('--dataprefix', default = 'data_', type = str, action = 'store', help = 'Prefix for the tfrecords')
    parser.add_argument('--checkmode', default = 0, type = int, action = 'store', help = 'Whether use the check mode')
    parser.add_argument('--checkexist', default = 1, type = int, action = 'store', help = 'Whether check the file exists in remote host')
    parser.add_argument('--checkblack', default = 0, type = int, action = 'store', help = 'Whether check the blackfiles')

    return parser

def get_jpglist(in_file):
    fin = open(in_file, 'r')
    all_lines = fin.readlines()

    all_list = []

    for each_line in all_lines:
        try:
            line_splits = each_line.split()
            jpg_name = line_splits[0]
            curr_label = int(line_splits[1])

            all_list.append((jpg_name, curr_label))
        except:
            print(each_line)

    return all_list

def main():

    parser = get_parser()
    args    = parser.parse_args()

    tfr_img_fld = os.path.join(args.savedir, 'image')
    tfr_lbl_fld = os.path.join(args.savedir, 'label')
    os.system('mkdir -p %s' % tfr_img_fld)
    os.system('mkdir -p %s' % tfr_lbl_fld)

    ssh_tfr_img_fld = os.path.join(args.sshfolder, 'image')
    ssh_tfr_lbl_fld = os.path.join(args.sshfolder, 'label')
    os.system('ssh %s mkdir -p %s' % (args.sshprefix, ssh_tfr_img_fld))
    os.system('ssh %s mkdir -p %s' % (args.sshprefix, ssh_tfr_lbl_fld))

    blacklist = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
               'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
               'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
               'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
               'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
               'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
               'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
               'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
               'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
               'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
               'n07583066_647.JPEG', 'n13037406_4650.JPEG', 'n02105855_2933.JPEG']


    for curr_indx in xrange(args.staindx, args.staindx + args.lenindx):
        txtname_now = '%s%i.txt' % (args.txtprefix, curr_indx)
        if not os.path.exists(txtname_now):
            continue

        name_list = get_jpglist(txtname_now)
        
        #print(name_list[:3])

        if args.checkmode==0:
            img_place = tf.placeholder(dtype=tf.uint8)
            img_raw = tf.image.encode_png(img_place)

        sess = tf.Session()

        tfrs_name = "%s_%i.tfrecords" % (args.dataprefix, curr_indx)
        image_tfrs_path = os.path.join(tfr_img_fld, tfrs_name)
        label_tfrs_path = os.path.join(tfr_lbl_fld, tfrs_name)

        remote_img_path = os.path.join(ssh_tfr_img_fld, tfrs_name)
        remote_lbl_path = os.path.join(ssh_tfr_lbl_fld, tfrs_name)

        if args.checkblack==1:
            black_flag = False
            for jpg_path, lbl in name_list:
                #print(jpg_path)
                if jpg_path.split('/')[-1] in blacklist:
                    black_flag = True
                    break
            if black_flag:
                os.system('ssh %s rm %s' % (args.sshprefix, remote_img_path))
                os.system('ssh %s rm %s' % (args.sshprefix, remote_lbl_path))

        if args.checkexist>=1:
            img_output = subprocess.check_output(['ssh', '-q', args.sshprefix, '[[ -f %s ]] && echo "Yes" || echo "No";' % remote_img_path])
            lbl_output = subprocess.check_output(['ssh', '-q', args.sshprefix, '[[ -f %s ]] && echo "Yes" || echo "No";' % remote_lbl_path])
            if 'Yes' in img_output and 'Yes' in lbl_output:
                print('%s already exists! Skip!' % tfrs_name)
                continue
            if args.checkexist==2:
                print('%s not exists!' % tfrs_name)
                continue

        if args.checkmode==0:
            image_writer = tf.python_io.TFRecordWriter(image_tfrs_path)
            label_writer = tf.python_io.TFRecordWriter(label_tfrs_path)
        else:
            image_iter = tf.python_io.tf_record_iterator(path=image_tfrs_path)
            label_iter = tf.python_io.tf_record_iterator(path=label_tfrs_path)

        for jpg_path, lbl in name_list:
            #print(jpg_path)
            if jpg_path.split('/')[-1] in blacklist:
                continue

            try:
                now_im = Image.open(jpg_path)
                new_im = now_im.resize((256, 256), Image.ANTIALIAS)
                img = np.array(new_im)
                img = img.astype(np.uint8)

                if len(img.shape)==2:
                    new_img = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                    new_img[:, :, :] = img[:, :, np.newaxis]
                    img = new_img
            except:
                print(jpg_path)
                continue

            if args.checkmode==0:
                img_raw_str = sess.run(img_raw, feed_dict = {img_place: img})
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': _bytes_feature(img_raw_str)}))
                image_writer.write(example.SerializeToString())

                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(lbl)}))
                label_writer.write(example.SerializeToString())
            else:
                img_string_record = image_iter.next()
                example = tf.train.Example()
                example.ParseFromString(img_string_record)
                img_string = (example.features.feature['image']
                                              .bytes_list
                                              .value[0])
                img_vector = tf.image.decode_png(img_string)

                img_array = sess.run(img_vector)

                #print(np.allclose(img, img_array))
                if not (np.allclose(img, img_array)):
                    print(jpg_path)

                lbl_string_record = label_iter.next()
                example = tf.train.Example()
                example.ParseFromString(lbl_string_record)
                lbl_decode = int(example.features.feature['label']
                                              .int64_list
                                              .value[0])
                #print(lbl_decode==lbl)
                if not (lbl_decode==lbl):
                    print(jpg_path)

        if args.checkmode==0:
            image_writer.close()
            label_writer.close()

            os.system('rsync -vzh %s %s:%s' % (image_tfrs_path, args.sshprefix, ssh_tfr_img_fld))
            os.system('rsync -vzh %s %s:%s' % (label_tfrs_path, args.sshprefix, ssh_tfr_lbl_fld))

if __name__=="__main__":
    main()
