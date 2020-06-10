import numpy as np
import os
import sys
import argparse
import pdb
import cv2
from tqdm import tqdm

from script_download import load_csv


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to downsample the videos')
    parser.add_argument(
            '--csvpath', default='/mnt/fs1/Dataset/kinetics/kinetics_train.csv', 
            type=str, action='store', 
            help='Path to the csv containing the information')
    parser.add_argument(
            '--dwnsmpldir', 
            default='/mnt/fs1/Dataset/kinetics/vd_dwnld_5fps', 
            type=str, action='store', 
            help='Directory to hold the tfrecords')
    parser.add_argument(
            '--avidir', 
            default='/mnt/fs1/Dataset/kinetics/vd_dwnld', 
            type=str, action='store', help='Directory to load the avis')
    parser.add_argument(
            '--len_avi', default=2000, type=int, action='store', 
            help='Number of avis processed')
    parser.add_argument(
            '--sta_avi', default=0, type=int, action='store', 
            help='Starting index of avis')
    parser.add_argument(
            '--check', default=1, type=int, action='store', 
            help='Whether checking the existence')
    parser.add_argument(
            '--newfps', default=5, type=int, action='store', 
            help='The new fps after downsampling')
    return parser


def main():
    parser = get_parser()

    args = parser.parse_args()
    os.system('mkdir -p %s' % args.dwnsmpldir)

    csv_data = load_csv(args.csvpath)

    start_indx_avi = args.sta_avi
    if args.len_avi > 0:
        end_indx_avi = min(args.sta_avi + args.len_avi, len(csv_data))
    else:
        end_indx_avi = len(csv_data)

    for curr_avi_indx in tqdm(xrange(start_indx_avi, end_indx_avi)):

        curr_dict = csv_data[curr_avi_indx]
        curr_avi_path = os.path.join(
                args.avidir, "vd_%i_mini.avi" % curr_dict['indx'])
        if not os.path.exists(curr_avi_path):
            continue

        out_avi_path = os.path.join(
                args.dwnsmpldir, "vd_%i_mini.avi" % curr_dict['indx'])
        if args.check==1:
            if os.path.exists(out_avi_path):
                continue

        vidcap = cv2.VideoCapture(curr_avi_path)
        # Get frame number, fps, height, width
        vid_len = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        vid_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vid_width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)

        vid_len = int(vid_len)
        if vid_len==0:
            continue

        # get all the frames
        frame_list = []
        for _ in xrange(vid_len):
            suc, frame = vidcap.read()
            frame_list.append(frame)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vidwrt = cv2.VideoWriter(
                out_avi_path, fourcc, args.newfps, 
                (int(vid_width), int(vid_height)))

        frm_idx = np.arange(0, vid_len, fps/args.newfps)

        for curr_indx in frm_idx:
            curr_indx = int(curr_indx)
            if curr_indx < len(frame_list):
                vidwrt.write(frame_list[curr_indx])

        vidwrt.release()
        vidcap.release()


if __name__ == '__main__':
    main()
