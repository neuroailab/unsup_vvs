import sys, os
import numpy as np
import argparse
from nltk.corpus import wordnet as wn

def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to subsample synsets of ImageNet, \
                    default parameters are for kanefsky')
    parser.add_argument(
            '--raw_dir', 
            default='/mnt/data3/chengxuz/Dataset/Imagenet_original', type=str, 
            action='store', help='Directory having the raw images')
    parser.add_argument(
            '--watch_depth', 
            default=2, type=int, 
            action='store', help='Depth of path to watch')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Get all synsets from train folder
    train_dir = os.path.join(args.raw_dir, 'train')
    all_synsets = os.listdir(train_dir)
    check_num = 10
    now_num = 0
    all_pr_synsets = []
    for each_synset in all_synsets:
        curr_syn = wn._synset_from_pos_and_offset('n', int(each_synset[1:]))
        curr_paths = curr_syn.hypernym_paths()
        #print(curr_syn, [curr_path[2] for curr_path in curr_paths])
        curr_names = np.concatenate(
                [
                    [each_path.name() for each_path in curr_path] 
                    for curr_path in curr_paths
                ], axis=0)
        #all_pr_synsets.extend([curr_path[args.watch_depth].name() for curr_path in curr_paths])
        have_sub = False
        #have_tot = False
        have_tot = True
        for each_name in curr_names:
            #if 'animal' in each_name or 'furniture' in each_name:
            #if 'furniture' in each_name:
            #if 'animal' in each_name:
            #if 'vehicle' in each_name:
            #if 'dog' in each_name:
            #if 'bear' in each_name:
            #if 'shoe' in each_name:
            #if 'car' in each_name:
            if 'television' in each_name:
                have_sub = True
                #all_pr_synsets.append(curr_syn)
                #break
            #if 'animal' in each_name:
            if 'vehicle' in each_name:
                have_tot = True
                #pass
        if have_sub and have_tot:
            #all_pr_synsets.append(curr_syn)
            all_pr_synsets.append(curr_names)

        if now_num>check_num:
            pass
            #break
        now_num = now_num +1
    print(len(all_pr_synsets))
    for each_names in all_pr_synsets:
        print(each_names)
    #print(len(np.unique(all_pr_synsets)))
    #print(np.unique(all_pr_synsets))

if __name__=='__main__':
    main()

#wn._synset_from_pos_and_offset
#hypernym_paths()
