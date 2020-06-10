import sys, os
import numpy as np
import scipy.io as sio

def make_cp_list(label_path, jpeg_folder, num_im = 50000):
    all_data = open(label_path, 'r')
    file_list = []
    label_list = []
    for indx_cat in xrange(num_im):
        curr_label = int(all_data.readline()) - 1
        label_list.append(curr_label)

        curr_file = os.path.join(jpeg_folder, 'ILSVRC2012_val_%08i.JPEG' % (indx_cat + 1))
        file_list.append(curr_file)

    return file_list, label_list

label_path = '/scratch/users/chengxuz/Data/imagenet_devkit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
jpeg_folder = '/scratch/users/chengxuz/Data/imagenet_val'

file_list, label_list = make_cp_list(label_path, jpeg_folder)
#print(file_list[:10])
#print(label_list[:10])

#exit()

all_len = 50000

num_tfr = 50

txt_prefix = '/scratch/users/chengxuz/Data/imagenet_devkit/val_fname_'

file_len = int(all_len/num_tfr)
now_findx = 0
for sta_ind in xrange(0, all_len, file_len):
    end_ind = min(sta_ind + file_len, all_len)

    now_txt = '%s%i.txt' % (txt_prefix, now_findx)

    fout = open(now_txt, 'w')
    for curr_indx in xrange(sta_ind, end_ind):
        fout.write('%s %i\n' % (file_list[curr_indx], label_list[curr_indx]))
    fout.close()
    now_findx = now_findx + 1
