import numpy as np
import os
import sys
import argparse
import pdb
import cv2

# Borrowed from https://github.com/nficano/pytube
from pytube import YouTube

def get_parser():
    parser = argparse.ArgumentParser(description='The script to download and save the videos from Youtube')
    parser.add_argument('--csvpath', default = '/mnt/fs1/Dataset/kinetics/kinetics_train.csv', type = str, action = 'store', help = 'Path to the csv containing the information')
    parser.add_argument('--savedir', default = '/mnt/fs1/Dataset/kinetics/vd_dwnld', type = str, action = 'store', help = 'Directory to hold the downloaded videos')
    parser.add_argument('--minresz', default = 256, type = int, action = 'store', help = 'Size for the smallest edge to be resized to')
    parser.add_argument('--sta_idx', default = 0, type = int, action = 'store', help = 'Start index for downloading')
    parser.add_argument('--len_idx', default = 2500, type = int, action = 'store', help = 'Length of index of downloading')
    parser.add_argument('--check', default = 0, type = int, action = 'store', help = 'Whether checking the existence')

    return parser

def load_csv(csvpath):
    fin = open(csvpath, 'r')
    csv_lines = fin.readlines()
    csv_lines = csv_lines[1:]
    all_data = []

    cate_list = []

    curr_indx = 0

    for curr_line in csv_lines:
        if curr_line[-1]=='\n':
            curr_line = curr_line[:-1]
        line_split = curr_line.split(',')
        curr_dict = {'cate': line_split[0], 'id': line_split[1], 'sta': int(line_split[2]), 'end': int(line_split[3]), 'train': line_split[4], 'flag': int(line_split[5]), 'indx': curr_indx}

        if not curr_dict['cate'] in cate_list:
            cate_list.append(curr_dict['cate'])

        curr_dict['cate_lbl'] = cate_list.index(curr_dict['cate'])

        all_data.append(curr_dict)
        curr_indx = curr_indx + 1

    return all_data

def main():
    parser = get_parser()

    args = parser.parse_args()

    os.system('mkdir -p %s' % args.savedir)

    csv_data = load_csv(args.csvpath)

    youtube_frmt = "http://youtu.be/%s"
    #for curr_indx, curr_data in enumerate(csv_data):
    #    assert curr_data['end'] - curr_data['sta'] == 10, "Length not 10!"
    
    curr_len = min(len(csv_data) - args.sta_idx, args.len_idx)

    for curr_indx in xrange(args.sta_idx, args.sta_idx + curr_len):
        curr_data = csv_data[curr_indx]
        curr_youtube_web = youtube_frmt % curr_data['id']
        try:
            yt = YouTube(curr_youtube_web)
        except:
            print('Not able to open!')
            continue
        curr_fname = 'vd_%i' % curr_indx

        avi_path = os.path.join(args.savedir, '%s_mini.avi' % curr_fname)
        print(avi_path)
        if os.path.exists(avi_path):
            if args.check==0:
                os.system('rm %s' % avi_path)
            else:
                continue

        yt.set_filename(curr_fname)
        #print(yt.get_videos())
        video = yt.get('mp4', '360p')

        mp4_path = os.path.join(args.savedir, '%s.mp4' % curr_fname)
        if os.path.exists(mp4_path):
            os.system('rm %s' % mp4_path)
        video.download(args.savedir)
        vidcap = cv2.VideoCapture(mp4_path)

        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
        start_indx = curr_data['sta'] * fps
        len_frame = (curr_data['end'] - curr_data['sta']) * fps
        vid_len = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        #print(vid_len)
        #print(start_indx)
        #print(len_frame)
        len_frame = int(min(vid_len - start_indx, len_frame))

        vid_height = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        vid_width = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)

        if vid_width < vid_height:
            vid_height = int(vid_height*1.0/vid_width*args.minresz)
            vid_width = 256
        else:
            vid_width = int(vid_width*1.0/vid_height*args.minresz)
            vid_height = 256

        fourcc = cv2.cv.CV_FOURCC(*'MP4V')
        vidwrt = cv2.VideoWriter(avi_path, fourcc, fps, (vid_width, vid_height))
        vidcap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, int(start_indx))

        #test_dir = os.path.join(args.savedir, 'vd_0')
        #os.system('mkdir -p %s' % test_dir)

        for indx_frame in xrange(len_frame):
            suc, frame = vidcap.read()
            #print(suc, indx_frame, len_frame)
            if not suc:
                #print(frame.shape)
                break
            frame_rsz = cv2.resize(frame, (vid_width, vid_height))
            vidwrt.write(frame_rsz)
            #cv2.imwrite(os.path.join(test_dir, 'im_%i.jpg' % indx_frame), frame_rsz)

        vidwrt.release()
        vidcap.release()

        os.system('rm %s' % mp4_path)

        if curr_indx%20==0:
            print(curr_indx)

if __name__ == '__main__':
    main()
