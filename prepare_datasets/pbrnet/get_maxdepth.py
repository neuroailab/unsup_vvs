import os,sys
from scipy import misc

depth_folder = '/scratch/users/chengxuz/Data/pbrnet/depth'

all_folders = os.listdir(depth_folder)
all_folders.sort()

#print(all_folders[:10])

print('Begin to count!')

max_value = 0

num_imgs = 0

for curr_folder in all_folders:
    curr_path = os.path.join(depth_folder, curr_folder)
    curr_img_list = os.listdir(curr_path)

    for curr_img_name in curr_img_list:

