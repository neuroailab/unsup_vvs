import tensorflow as tf
import cv2
import numpy as np
import os

from write_tfrs import _bytes_feature

def get_frames(vd_path, crop_time = 5, crop_rate = 5, crop_hei = 224, crop_wid = 224, replace_folder = None):

    if not replace_folder is None:
        base_vd_path = os.path.basename(vd_path)
        vd_path = os.path.join(replace_folder, base_vd_path)

    vidcap = cv2.VideoCapture(vd_path)
    vid_len = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    fps = int(fps)
    vid_height = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    vid_width = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)

    vid_len = int(vid_len)

    #print(vd_path, vid_len, fps, vid_height, vid_width)

    #print(vid_len, fps, curr_avi_path)

    # get all the frames
    frame_list = []
    for _ in xrange(vid_len):
        suc, frame = vidcap.read()
        frame_list.append(frame)

    for _ in xrange(vid_len, fps*min(crop_time, 10)):
        frame_list.append(np.zeros(frame_list[0].shape, dtype = np.uint8))

    frame_len = len(frame_list)
    frame_list = np.asarray(frame_list)

    start_indx = np.random.randint(frame_len - crop_time * fps + 1)
    frame_list = frame_list[start_indx:start_indx + crop_time * fps]
    
    if fps%crop_rate==0:
        frame_list = frame_list[::(fps//crop_rate)]
    else:
        want_len = crop_time * crop_rate
        curr_len = crop_time * fps
        if want_len <= curr_len:
            indx_choice = np.random.choice(curr_len, want_len, replace = False)
        else:
            indx_choice = np.random.choice(curr_len, want_len % curr_len, replace = False)
            indx_choice = np.concatenate([range(curr_len) * (want_len/curr_len), indx_choice])
        indx_choice.sort()
        frame_list = frame_list[indx_choice]

    # Do random crop here
    hei_sta = np.random.randint(vid_height - crop_hei)
    wid_sta = np.random.randint(vid_width - crop_wid)
    frame_list = frame_list[:, hei_sta:hei_sta + crop_hei, wid_sta:wid_sta + crop_wid, :]
    #print(vd_path, vid_len, fps, frame_list.shape)
    vidcap.release()

    return frame_list


if __name__ == '__main__':
    test_str = '/home/chengxuz/vd_0_mini.avi'
    tfr_path = '/home/chengxuz/str_test.tfrecords'

    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    writer = tf.python_io.TFRecordWriter(tfr_path)

    example = tf.train.Example(features=tf.train.Features(feature={
        'path': _bytes_feature(test_str)}))
    writer.write(example.SerializeToString())
    writer.close()

    record_iterator = tf.python_io.tf_record_iterator(path=tfr_path)

    path_place = tf.placeholder(dtype=tf.string)
    #frames_ten = tf.py_func(lambda vd_path: get_frames(vd_path, 5, 5), [path_place], tf.uint8)
    frames_ten = tf.py_func(lambda vd_path: get_frames(vd_path, 5, 5, replace_folder = '/data2/chengxuz/kinetics/vd_dwnld_val/'), [path_place], tf.uint8)

    sess = tf.Session()

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        img_string = (example.features.feature['path']
                                      .bytes_list
                                      .value[0])
        print(img_string)

        #frames = get_frames(img_string)
        #print(frames.shape)

        #frames_from_tf = sess.run(frames_ten, feed_dict = {path_place: img_string})
        frames_from_tf = sess.run(frames_ten, feed_dict = {path_place: '/mnt/fs1/Dataset/kinetics/vd_dwnld_val/vd_7542_mini.avi'})
        print(frames_from_tf.shape)

