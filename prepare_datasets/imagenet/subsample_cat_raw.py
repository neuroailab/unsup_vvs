import os
import argparse
import sys
import numpy as np
import inter_choose_nouns

def get_parser():
    parser = argparse.ArgumentParser(
            description='Randomly sample some categories from raw ImageNet ' \
                    + 'and build links in new folder')
    parser.add_argument(
            '--output_dir', 
            default='/data5/chengxuz/new_imagenet_raw_ctl',
            type=str, action='store', help='Directory to save the tfrecords')
    parser.add_argument(
            '--raw_dir', 
            default='/data5/chengxuz/Dataset/imagenet_raw', 
            type=str, action='store', 
            help='Folder having raw ImageNet style images')
    parser.add_argument(
            '--seed', 
            default=0, type=int, action='store', 
            help='Random seed for numpy')
    parser.add_argument(
            '--num', default=246, 
            type=int, action='store', help='Number of categories to sample')
    parser.add_argument(
            '--output_txt', 
            default='./infant_noun_ImageNet_addmore.txt', type=str, 
            action='store', help='Txt file storing the results')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    new_train_dir = os.path.join(args.output_dir, 'train')
    new_val_dir = os.path.join(args.output_dir, 'val')
    os.system('mkdir -p %s' % args.output_dir)
    os.system('mkdir -p %s' % new_train_dir)
    os.system('mkdir -p %s' % new_val_dir)
    # Exclude the categories included in infant imagenet
    _, all_synsets = inter_choose_nouns.load_curr_results(args)
    train_dir = os.path.join(args.raw_dir, 'train')
    val_dir = os.path.join(args.raw_dir, 'val')
    raw_synsets = os.listdir(train_dir)
    raw_synsets = filter(lambda x: x not in all_synsets, raw_synsets)
    # sample args.num categories
    np.random.seed(0)
    sampled_synsets = np.random.choice(raw_synsets, args.num, replace=False)
    for each_synset in sampled_synsets:
        def _make_ln(synset, old_dir, new_dir):
            os.system('ln -s %s %s' \
                    % (os.path.join(old_dir, synset),\
                       os.path.join(new_dir, synset)))
        _make_ln(each_synset, train_dir, new_train_dir)
        _make_ln(each_synset, val_dir, new_val_dir)

if __name__=="__main__":
    main()
