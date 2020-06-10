import sys, os
import numpy as np
import argparse
import parse_infant_nouns

def get_parser():
    parser = argparse.ArgumentParser(
            description='Interactively subsample synsets of ImageNet, \
                    default parameters are for kanefsky')
    parser.add_argument(
            '--raw_dir', 
            default='/mnt/data3/chengxuz/Dataset/Imagenet_original', type=str, 
            action='store', help='Directory having the raw images')
    parser.add_argument(
            '--output_txt', 
            default='./infant_noun_ImageNet.txt', type=str, 
            action='store', help='Txt file storing the results')
    return parser

def get_synset_dict(args):
    train_dir = os.path.join(args.raw_dir, 'train')
    all_synsets = os.listdir(train_dir)
    synset_dict = {}
    for each_synset in all_synsets:
        curr_syn = wn._synset_from_pos_and_offset('n', int(each_synset[1:]))
        curr_paths = curr_syn.hypernym_paths()
        curr_names = np.concatenate(
                [
                    [each_path.name() for each_path in curr_path] 
                    for curr_path in curr_paths
                ], axis=0)
        curr_names_str = curr_names[0]
        for left_name in curr_names[1:]:
            curr_names_str += '-%s' % left_name
        synset_dict[each_synset] = {
                'synset':curr_syn,
                'name_str':each_synset,
                'paths_str':curr_names_str,
                }
    return synset_dict

def load_curr_results(args, ret_dict=False):
    if not os.path.isfile(args.output_txt):
        return [], []
    fin = open(args.output_txt, 'r')
    all_lines = fin.readlines()
    all_nouns = []
    all_synsets = []
    noun_to_synset = {}
    for each_line in all_lines:
        each_line = each_line[:-1]
        # each line is like "noun: synset_0 synset_1 synset_2 ..."
        curr_split = each_line.split(':')
        curr_noun = curr_split[0]
        all_nouns.append(curr_noun)
        curr_synsets = curr_split[1].split(' ')
        noun_to_synset[curr_noun] = []
        for each_synset in curr_synsets:
            # Avoid empty split here
            if each_synset.startswith('n'):
                all_synsets.append(each_synset)
                noun_to_synset[curr_noun].append(each_synset)
    fin.close()
    if ret_dict:
        return noun_to_synset
    else:
        return all_nouns, all_synsets

def query_relate_synset(new_key, curr_paths):
    if isinstance(new_key, str):
        return new_key in curr_paths
    else:
        assert isinstance(new_key, list)
        return new_key[0] in curr_paths and new_key[1] in curr_paths

def main():
    from nltk.corpus import wordnet as wn
    parser = get_parser()
    args = parser.parse_args()
    # Get all synsets and their paths from train folder
    synset_dict = get_synset_dict(args)
    # Get infant nouns
    noun_dict = parse_infant_nouns.load_noun_dict()
    done_nouns, done_synsets = load_curr_results(args)
    for done_noun in done_nouns:
        noun_dict.pop(done_noun)
    for done_synset in done_synsets:
        synset_dict.pop(done_synset)
    keys_of_inter = [each_key for each_key in noun_dict \
            if noun_dict[each_key]['second_big']==0]
    keys_of_inter.sort()
    num_koi = len(keys_of_inter)
    finished = 0
    for each_key in keys_of_inter:
        fout = open(args.output_txt, 'a+')
        while True:
            # Process one key
            print('Current noun: %s' % each_key, \
                  '%d remaining' % (num_koi - finished))
            init_command = raw_input()
            ## Change the search key if needed
            if init_command == 'c':
                new_key = each_key
            elif init_command == 'd':
                fout.write('%s: \n' % each_key)
                fout.close()
                finished += 1
                break
            elif init_command == 'a':
                new_key = [each_key]
                added_noun = raw_input()
                new_key.append(added_noun)
            else:
                new_key = init_command
            write_str = '%s:' % each_key
            add_synsets = []
            add_names = []
            relate_synsets = []
            for each_synset in synset_dict.keys():
                curr_paths = synset_dict[each_synset]['paths_str']
                if query_relate_synset(new_key, curr_paths):
                    relate_synsets.append(each_synset)
            keep_all = False
            for curr_id, each_synset in enumerate(relate_synsets):
                print('Current synset:', \
                        synset_dict[each_synset]['synset'], \
                        '[%d / %d]' % (curr_id, len(relate_synsets)))
                print(synset_dict[each_synset]['paths_str'])
                if not keep_all:
                    command = raw_input()
                if command == 'keep all':
                    keep_all = True
                if command == 'c' or keep_all:
                    ## Add the synset to this key
                    add_synsets.append(each_synset)
                    add_names.append(synset_dict[each_synset]['synset'])
                    write_str += ' %s' % each_synset
            print('End of synsets,', add_names)
            write_str += '\n'
            # Write the result to file if wanted
            end_command = raw_input()
            if end_command == 'c':
                finished += 1
                fout.write(write_str)
                ## Refresh the writer
                fout.close()
                ## Delete the synset already added
                for each_synset in add_synsets:
                    synset_dict.pop(each_synset)
                break

if __name__=='__main__':
    main()
