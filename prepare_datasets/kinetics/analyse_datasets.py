import numpy as np
import os
import sys
import argparse
import pdb
import cv2

from script_download import load_csv

def get_parser():
    parser = argparse.ArgumentParser(description='The script to download and save the videos from Youtube')
    parser.add_argument('--csvpath', default = '/mnt/fs1/Dataset/kinetics/kinetics_train.csv', type = str, action = 'store', help = 'Path to the csv containing the information')
    parser.add_argument('--avidir', default = '/data2/chengxuz/kinetics/vd_dwnld', type = str, action = 'store', help = 'Directory to load the avis')

    return parser

def main():
    parser = get_parser()

    args = parser.parse_args()

    csv_data = load_csv(args.csvpath)

    non_num = 0
    zero_num = 0

    #all_stat = {}
    all_fps = []

    for curr_indx in xrange(len(csv_data)):
        curr_dict = csv_data[curr_indx]
        curr_avi_path = os.path.join(args.avidir, "vd_%i_mini.avi" % curr_dict['indx'])
        if not os.path.exists(curr_avi_path):
            non_num = non_num + 1
            continue
        vidcap = cv2.VideoCapture(curr_avi_path)

        # Get frame number, fps, height, width
        vid_len = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
        vid_height = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        vid_width = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)

        all_fps.append(fps)

        if int(vid_len)==0:
            zero_num = zero_num + 1

        if curr_indx % 1000==0:
            print(curr_indx)
            all_fps_na = np.asarray(all_fps)
            unq_fps = np.unique(all_fps_na)
            nm_fps_dt = [(v, np.sum(all_fps_na == v)) for v in unq_fps]
            nm_fps_dt = sorted(nm_fps_dt, key = lambda v: -v[1])
            '''
            for curr_fps, curr_nm in nm_fps_dt:
                print '%.2f: %i,' % (curr_fps, curr_nm),
            #print(nm_fps_dt)
            print
            '''

    print(non_num)
    print(zero_num)

if __name__ == '__main__':
    main()
