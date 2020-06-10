import sys, os
import numpy as np
import argparse
import inter_choose_nouns
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(
            description='Building new ImageNet from subsampled synsets, ' +\
                    'default parameters are for node07')
    parser.add_argument(
            '--add_dir', 
            default='/data5/chengxuz/imagenet_additional_tar/', type=str, 
            action='store', help='Directory to save the images')
    parser.add_argument(
            '--raw_dir', 
            default='/data5/chengxuz/Dataset/imagenet_raw', type=str, 
            action='store', help='Directory having the raw images')
    parser.add_argument(
            '--output_txt', 
            default='./infant_noun_ImageNet_addmore.txt', type=str, 
            action='store', help='Txt file storing the results')
    parser.add_argument(
            '--output_dir', 
            default='/data5/chengxuz/new_imagenet_raw', type=str, 
            action='store', help='Output directory')
    parser.add_argument(
            '--random_seed', 
            default=0, type=int, 
            action='store', help='Random seed for numpy')
    parser.add_argument(
            '--max_img_num', 
            default=1300, type=int,
            action='store', help='Maximal number to have')
    parser.add_argument(
            '--val_num', 
            default=50, type=int,
            action='store', 
            help='Number of images in validation for each synset')
    return parser

def get_imgs_from_dir(synset_dir):
    curr_imgs = os.listdir(synset_dir)
    curr_imgs = [os.path.join(synset_dir, each_img) 
            for each_img in curr_imgs]
    curr_imgs.sort()
    return curr_imgs

def make_links(curr_dir, all_imgs):
    os.system('mkdir -p %s' % curr_dir)
    for source_file in all_imgs:
        dest_file = os.path.join(curr_dir, os.path.basename(source_file))
        os.system('ln -s %s %s' % (source_file, dest_file))

def main():
    parser = get_parser()
    args = parser.parse_args()
    os.system('mkdir -p %s' % args.output_dir)
    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    os.system('mkdir -p %s' % train_dir)
    os.system('mkdir -p %s' % val_dir)
    # Get the noun to synset dict
    noun_to_synset = inter_choose_nouns.load_curr_results(
            args, ret_dict=True)
    raw_synsets = os.listdir(os.path.join(args.raw_dir, 'train'))
    add_synsets = os.listdir(args.add_dir)
    # Create links for each noun
    np.random.seed(0)
    for each_noun, all_synsets in tqdm(noun_to_synset.items()):
        is_raw = False
        is_add = False
        all_imgs = []
        all_val_imgs = []
        for each_synset in all_synsets:
            is_raw = is_raw or each_synset in raw_synsets
            is_add = is_add or each_synset in add_synsets
            if each_synset in raw_synsets:
                synset_dir = os.path.join(args.raw_dir, 'train', each_synset)
                curr_val_dir = os.path.join(args.raw_dir, 'val', each_synset)
                all_val_imgs.extend(get_imgs_from_dir(curr_val_dir))
            else:
                synset_dir = os.path.join(args.add_dir, each_synset)
            all_imgs.extend(get_imgs_from_dir(synset_dir))
        if is_raw:
            train_imgs = np.random.choice(
                    all_imgs, 
                    min(args.max_img_num, len(all_imgs)),
                    replace=False,
                    )
            val_imgs = np.random.choice(
                    all_val_imgs,
                    min(args.val_num, len(all_val_imgs)),
                    replace=False,
                    )
        else:
            train_val_imgs = np.random.choice(
                    all_imgs, 
                    min(args.max_img_num + args.val_num, len(all_imgs)),
                    replace=False,
                    )
            val_imgs = train_val_imgs[:args.val_num]
            train_imgs = train_val_imgs[args.val_num:]
        new_noun_name = each_noun.replace(' ', '_')
        make_links(os.path.join(train_dir, new_noun_name), train_imgs)
        make_links(os.path.join(val_dir, new_noun_name), val_imgs)

if __name__=='__main__':
    main()
