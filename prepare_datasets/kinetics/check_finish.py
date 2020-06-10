import numpy as np
import os
import sys
import argparse
import pdb
import cv2

from pytube import YouTube
from script_download import load_csv

def get_parser():
    parser = argparse.ArgumentParser(description='The script to download and save the videos from Youtube')
    parser.add_argument('--csvpath', default = '/data/chengxuz/kinetics/kinetics_train/kinetics_train.csv', type = str, action = 'store', help = 'Path to the csv containing the information')
    parser.add_argument('--savedir', default = '/mnt/fs1/Dataset/kinetics/vd_dwnld', type = str, action = 'store', help = 'Directory to hold the downloaded videos')

    return parser

def main():
    parser = get_parser()

    args = parser.parse_args()

    csv_data = load_csv(args.csvpath)

    lose_num = 0
    youtube_frmt = "http://youtu.be/%s"

    for curr_indx in xrange(len(csv_data)):
        curr_fname = 'vd_%i' % curr_indx

        if curr_indx%500==0:
            print(curr_indx)

        avi_path = os.path.join(args.savedir, '%s_mini.avi' % curr_fname)
        if os.path.exists(avi_path):
            continue
        else:
            lose_num = lose_num + 1
            print(avi_path, lose_num)
            #break

        curr_data = csv_data[curr_indx]
        curr_youtube_web = youtube_frmt % curr_data['id']
        try:
            yt = YouTube(curr_youtube_web)
        except:
            print('Not able to open!')

if __name__ == '__main__':
    main()
