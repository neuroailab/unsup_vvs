import numpy as np
import os
import sys
import argparse
import pdb
import cv2

from script_download import load_csv
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_parser():
    parser = argparse.ArgumentParser(description='The script to download and save the videos from Youtube')
    parser.add_argument('--csvpath', default = '/mnt/fs1/Dataset/kinetics/kinetics_train.csv', type = str, action = 'store', help = 'Path to the csv containing the information')
    parser.add_argument('--tfrdir', default = '/mnt/fs1/Dataset/kinetics/train_tfrs', type = str, action = 'store', help = 'Directory to hold the tfrecords')
    parser.add_argument('--avidir', default = '/mnt/fs1/Dataset/kinetics/vd_dwnld', type = str, action = 'store', help = 'Directory to load the avis')
    parser.add_argument('--tfr_idx', default = 0, type = int, action = 'store', help = 'Start index for tfrecords')
    parser.add_argument('--len_tfr', default = 5, type = int, action = 'store', help = 'Number of tfrecords needed')
    parser.add_argument('--len_avi', default = 450, type = int, action = 'store', help = 'Number of avis needed in one tfrecord')
    parser.add_argument('--check', default = 0, type = int, action = 'store', help = 'Whether checking the existence')
    parser.add_argument('--seed', default = 0, type = int, action = 'store', help = 'Random seed for shuffling')
    parser.add_argument('--write_avi_path', default = 0, type = int, action = 'store', help = 'Whether writing the avi path, rather than frames')

    return parser

def get_avis(tfr_indx, csv_data, args):
    start_avi_indx = tfr_indx * args.len_avi
    end_avi_indx = start_avi_indx + args.len_avi

    ret_avis = []

    for curr_avi_indx in xrange(start_avi_indx, min(end_avi_indx, len(csv_data))):
        curr_dict = csv_data[curr_avi_indx]
        curr_avi_path = os.path.join(args.avidir, "vd_%i_mini.avi" % curr_dict['indx'])
        if os.path.exists(curr_avi_path):
            ret_avis.append([curr_avi_path, curr_dict['cate_lbl']])

    return ret_avis

def main():
    parser = get_parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()
    os.system('mkdir -p %s' % args.tfrdir)
    if args.write_avi_path==0:
        tfr_key_list = ['video', 'height', 'width', 'label']
    else:
        tfr_key_list = ['path', 'height_p', 'width_p', 'label_p']

    for curr_key in tfr_key_list:
        os.system('mkdir -p %s' % os.path.join(args.tfrdir, curr_key))

    csv_data = load_csv(args.csvpath)
    np.random.seed(args.seed)
    np.random.shuffle(csv_data)

    img_place = tf.placeholder(dtype=tf.uint8)
    img_raw = tf.image.encode_png(img_place)
    sess = tf.Session()

    for tfr_indx_rela in xrange(args.len_tfr):
        tfr_indx = args.tfr_idx + tfr_indx_rela

        avi_paths = get_avis(tfr_indx, csv_data, args)
        if len(avi_paths)==0:
            continue

        if args.write_avi_path==0:
            video_tfr_path = os.path.join(args.tfrdir, 'video', 'data_%i.tfrecords' % tfr_indx)
            height_tfr_path = os.path.join(args.tfrdir, 'height', 'data_%i.tfrecords' % tfr_indx)
            width_tfr_path = os.path.join(args.tfrdir, 'width', 'data_%i.tfrecords' % tfr_indx)
            label_tfr_path = os.path.join(args.tfrdir, 'label', 'data_%i.tfrecords' % tfr_indx)
        else:
            path_tfr_path = os.path.join(args.tfrdir, 'path', 'data_%i.tfrecords' % tfr_indx)
            height_tfr_path = os.path.join(args.tfrdir, 'height_p', 'data_%i.tfrecords' % tfr_indx)
            width_tfr_path = os.path.join(args.tfrdir, 'width_p', 'data_%i.tfrecords' % tfr_indx)
            label_tfr_path = os.path.join(args.tfrdir, 'label_p', 'data_%i.tfrecords' % tfr_indx)

        if args.check==1:
            if os.path.exists(video_tfr_path) and os.path.getsize(video_tfr_path) > 5349186072:
                continue

        # Define all the writers
        if args.write_avi_path==0:
            video_writer = tf.python_io.TFRecordWriter(video_tfr_path)
        else:
            path_writer = tf.python_io.TFRecordWriter(path_tfr_path)

        height_writer = tf.python_io.TFRecordWriter(height_tfr_path)
        width_writer = tf.python_io.TFRecordWriter(width_tfr_path)
        label_writer = tf.python_io.TFRecordWriter(label_tfr_path)

        # Write to tfrecords
        for curr_avi_path, curr_label in avi_paths:
            vidcap = cv2.VideoCapture(curr_avi_path)

            # Get frame number, fps, height, width
            vid_len = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
            vid_height = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
            vid_width = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)

            vid_len = int(vid_len)
            if vid_len==0:
                continue

            if args.write_avi_path==0:

                # get all the frames
                frame_list = []
                for _ in xrange(vid_len):
                    suc, frame = vidcap.read()
                    frame_list.append(frame)

                for _ in xrange(vid_len, int(fps*10)):
                    frame_list.append(np.zeros(frame_list[0].shape))

                img_idx_list = np.asarray(range(len(frame_list)))
                frame_len = len(frame_list)
                if frame_len > 250:
                    img_idx_list = np.random.choice(frame_len, 250, replace = False)

                if frame_len < 250:
                    add_img_idx_list = np.random.choice(frame_len, 250 % frame_len, replace = False)
                    img_idx_list = np.concatenate([range(len(frame_list)) * (250/frame_len), add_img_idx_list])
                img_idx_list.sort()

                #print('len ', len(frame_list))
                
                for curr_img_idx in img_idx_list:
                    #print(curr_img_idx)

                    img_raw_str = sess.run(img_raw, feed_dict = {img_place: frame_list[curr_img_idx]})

                    video_example = tf.train.Example(features=tf.train.Features(feature={
                        'video': _bytes_feature(img_raw_str)}))
                    video_writer.write(video_example.SerializeToString())

                    height_example = tf.train.Example(features=tf.train.Features(feature={
                        'height': _int64_feature(int(vid_height))}))
                    height_writer.write(height_example.SerializeToString())

                    width_example = tf.train.Example(features=tf.train.Features(feature={
                        'width': _int64_feature(int(vid_width))}))
                    width_writer.write(height_example.SerializeToString())

                    label_example = tf.train.Example(features=tf.train.Features(feature={
                        'label': _int64_feature(curr_label)}))
                    label_writer.write(label_example.SerializeToString())
            else:
                path_example = tf.train.Example(features=tf.train.Features(feature={
                    'path': _bytes_feature(curr_avi_path)}))
                path_writer.write(path_example.SerializeToString())

                height_example = tf.train.Example(features=tf.train.Features(feature={
                    'height_p': _int64_feature(int(vid_height))}))
                height_writer.write(height_example.SerializeToString())

                width_example = tf.train.Example(features=tf.train.Features(feature={
                    'width_p': _int64_feature(int(vid_width))}))
                width_writer.write(height_example.SerializeToString())

                label_example = tf.train.Example(features=tf.train.Features(feature={
                    'label_p': _int64_feature(curr_label)}))
                label_writer.write(label_example.SerializeToString())

        if args.write_avi_path==0:
            video_writer.close()
        else:
            path_writer.close()
        height_writer.close()
        width_writer.close()
        label_writer.close()

if __name__ == '__main__':
    main()
