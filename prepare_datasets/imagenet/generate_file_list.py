import sys, os
import numpy as np
import scipy.io as sio

def load_dict(meta_path, num_cat = 1000):
    data = sio.loadmat(meta_path)
    label_dict = {}
    label_list = []
    for indx_cat in xrange(num_cat):
        curr_synset = data['synsets'][indx_cat][0][1][0]
        curr_label = int(data['synsets'][indx_cat][0][0][0][0]) - 1
        label_dict[curr_synset] = curr_label
        label_list.append(curr_synset)

    return label_dict, label_list

meta_path = '/scratch/users/chengxuz/Data/imagenet_devkit/ILSVRC2012_devkit_t12/data/meta.mat'
home_dir_path = '/scratch/users/chengxuz/Data/imagenet_dir'

label_dict, label_list = load_dict(meta_path)
print(len(label_dict))

'''
all_dirs = os.listdir(home_dir_path)
in_num = 0
for curr_synset in label_dict:
    if curr_synset in all_dirs:
        in_num = in_num + 1

print(in_num)
'''

all_file_list = []
for curr_label in label_list:
    #print(curr_label)
    curr_file_list = os.listdir(os.path.join(home_dir_path, curr_label))
    curr_file_list.sort()

    for curr_file in curr_file_list:
        all_file_list.append((os.path.join(home_dir_path, curr_label, curr_file), label_dict[curr_label]))

#print(len(all_file_list))

np.random.seed(0)
new_indx_list = np.random.permutation(len(all_file_list))
new_file_list = []
for curr_indx in new_indx_list:
    new_file_list.append(all_file_list[curr_indx])
print(len(new_file_list))
print(new_file_list[0])
print(new_file_list[1])
print(all_file_list[0])

all_len = len(new_file_list)

num_tfr = 500

txt_prefix = '/scratch/users/chengxuz/Data/imagenet_devkit/fname_'

file_len = int(all_len/num_tfr)
now_findx = 0
for sta_ind in xrange(0, all_len, file_len):
    end_ind = min(sta_ind + file_len, all_len)

    now_txt = '%s%i.txt' % (txt_prefix, now_findx)

    fout = open(now_txt, 'w')
    for curr_indx in xrange(sta_ind, end_ind):
        fout.write('%s %i\n' % (new_file_list[curr_indx][0], new_file_list[curr_indx][1]))
    fout.close()
    now_findx = now_findx + 1
