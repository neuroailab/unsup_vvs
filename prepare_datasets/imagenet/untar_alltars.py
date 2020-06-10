import sys, os

home_path = '/scratch/users/chengxuz/Data/imagenet'
home_dir_path = '/scratch/users/chengxuz/Data/imagenet_dir'
all_tars = os.listdir(home_path)
all_tars = filter(lambda x: x.startswith('n'), all_tars)

os.system('mkdir -p %s' % home_dir_path)

print(len(all_tars))

for each_tar in all_tars:
    tar_path = os.path.join(home_path, each_tar)
    folder_path = os.path.join(home_dir_path, each_tar[:-4])

    print(tar_path, folder_path)

    os.system('mkdir -p %s' % folder_path)
    os.system('tar xf %s -C %s' % (tar_path, folder_path))

    #break
