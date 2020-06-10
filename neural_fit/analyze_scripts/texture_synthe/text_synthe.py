import numpy as np
from oct2py import Oct2Py
import pdb
oc = Oct2Py()

#load_file = '/mnt/fs4/chengxuz/v4it_temp_results/optimal_stimuli_examples/dc_v4.npy'
#save_file = '/mnt/fs4/chengxuz/v4it_temp_results/optimal_stimuli_examples/texture_synthe_results/dc_v4.npy'

#load_file = '/mnt/fs4/chengxuz/v4it_temp_results/optimal_stimuli_examples/cate_v4.npy'
#save_file = '/mnt/fs4/chengxuz/v4it_temp_results/optimal_stimuli_examples/texture_synthe_results/cate_v4.npy'

#load_file = '/mnt/fs4/chengxuz/v4it_temp_results/optimal_stimuli_examples/la_v4.npy'
#save_file = '/mnt/fs4/chengxuz/v4it_temp_results/optimal_stimuli_examples/texture_synthe_results/la_v4.npy'

#load_file = '/mnt/fs4/chengxuz/v4it_temp_results/optimal_stimuli_examples/color_v4.npy'
#save_file = '/mnt/fs4/chengxuz/v4it_temp_results/optimal_stimuli_examples/texture_synthe_results/color_v4.npy'

load_file = '/mnt/fs4/chengxuz/v4it_temp_results/optimal_stimuli_examples/dc_v4_all.npy'
save_file = '/mnt/fs4/chengxuz/v4it_temp_results/optimal_stimuli_examples/texture_synthe_results/dc_v4_all.npy'

orig_imgs = np.load(load_file)

new_imgs = []
for orig_img_idx in range(len(orig_imgs)):
    orig_img = orig_imgs[orig_img_idx]
    try:
        new_imgs.append(oc.func_syn(orig_img))
    except:
        new_imgs.append(orig_img)

new_imgs = np.stack(new_imgs, axis=0)
np.save(save_file, new_imgs)
