import os
import sys
from urllib2 import urlopen
import numpy as np
import requests
from multiprocessing import Pool
import argparse
import inter_choose_nouns
from tqdm import tqdm

def get_one_file(url_and_path):
    curr_url, filename = url_and_path
    curr_url = (curr_url)
    try:
        img_data = requests.get(curr_url, stream=True).content
        with open(filename, 'wb') as handler:
            handler.write(img_data)
            # validate image before moving on
            filesize = os.stat(filename).st_size
            # smaller than some threshold filesize
            if filesize<100000: 
                os.remove(filename)
                return 0
            else:
                return 1
    except Exception as e:
        pass
        #print e
    return 0

def get_urls_in_synset(
        synset, imagenet_username='jefan', 
        accesskey='f5f789c3fb79bfc5e76237ac3feb55b4e959b0ff'):
    url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?' + \
            'wnid=' + synset + \
            '&username=' + imagenet_username + \
            '&accesskey=' + accesskey + \
            '&release=latest'
    url_file = urlopen(url)
    all_urls = url_file.readlines()
    all_urls.sort()
    return all_urls

def download_tar_by_synset(
        synset,
        store_dir='/data5/chengxuz/imagenet_additional/',
        ):
    url = '"http://image-net.org/download/synset?wnid=%s' % synset + \
            '&username=jefan' + \
            '&accesskey=f5f789c3fb79bfc5e76237ac3feb55b4e959b0ff' + \
            '&release=latest&src=stanford"'
    output_path = os.path.join(store_dir, "%s.tar" % synset)
    os.system('wget -c %s -O %s' % (url, output_path))
    folder_path = os.path.join(store_dir, synset)
    os.system('mkdir -p %s' % folder_path)
    os.system('tar -xvf %s -C %s' % (output_path, folder_path))
    os.system('rm %s' % output_path)

def download_images_by_synset(
        synset, all_urls=None,
        store_dir='/data5/chengxuz/imagenet_additional/',
        ):
    """
    Downloads images by synset, like it says. 
    Takes in a list of synsets, and optionally number of photos per synset, 
    and saves images in a directory called photos 
    """
    thread_p = Pool(20)
    path = os.path.join(store_dir, synset)
    os.system('mkdir -p %s' % path)
    # Build url and paths and send them to the thread function
    if not all_urls:
        all_urls = get_urls_in_synset(synset)
    url_and_paths = []
    for now_idx, curr_url in enumerate(all_urls):
        curr_path = os.path.join(
                path,
                synset + '_{0:04d}.jpg'.format(now_idx))
        url_and_paths.append((curr_url, curr_path))
    dwn_res = thread_p.map(get_one_file, url_and_paths)
    print(synset, np.sum(np.asarray(dwn_res)))

def get_parser():
    parser = argparse.ArgumentParser(
            description='Download images for synsets of ImageNet, \
                    default parameters are for node07')
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
            '--syn_sta', 
            default=0, type=int, 
            action='store', help='Start synset index')
    parser.add_argument(
            '--syn_len', 
            default=1, type=int, 
            action='store', help='Number of synsets to download')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if False:
        # Count number of images in each noun
        noun_to_synset = inter_choose_nouns.load_curr_results(
                args, ret_dict=True)
        raw_synsets = os.listdir(os.path.join(args.raw_dir, 'train'))
        add_synsets = os.listdir(args.add_dir)
        noun_and_num = []
        for each_noun, all_synsets in noun_to_synset.items():
            curr_num = 0
            raw_flag = False
            for each_synset in all_synsets:
                if each_synset in raw_synsets:
                    raw_flag = True
                    break
                else:
                    if each_synset not in add_synsets:
                        print("Downloading for %s not finished!" % each_synset)
                    else:
                        curr_num += len(os.listdir(
                            os.path.join(args.add_dir, each_synset)
                            ))
            if raw_flag:
                continue
            else:
                noun_and_num.append((each_noun, curr_num, all_synsets))
        all_num_images = (len(noun_to_synset)-len(noun_and_num)) * 1300
        noun_and_num = sorted(noun_and_num, key=lambda x: x[1])
        for each_noun, curr_num, all_synsets in noun_and_num:
            print("%s has %i images" % (each_noun, curr_num))
            all_num_images += min(1300, curr_num)
        print("Overall: %i images, %i categories" % \
                (all_num_images, len(noun_to_synset)))

    # Get all synsets
    _, all_synsets = inter_choose_nouns.load_curr_results(args)
    # Get synsets to download
    raw_synsets = os.listdir(os.path.join(args.raw_dir, 'train'))
    os.system('mkdir -p %s' % args.add_dir)
    add_synsets = os.listdir(args.add_dir)
    dwnld_synsets = []
    for each_synset in all_synsets:
        if each_synset not in raw_synsets:
            dwnld_synsets.append(each_synset)
    dwnld_synsets.sort()
    # Check if there are duplicate urls
    url_to_synset = {}
    all_urls = []
    for each_synset in tqdm(dwnld_synsets):
        now_urls = get_urls_in_synset(each_synset)
        for each_url in now_urls:
            url_to_synset[each_url] = each_synset
        all_urls.extend(now_urls)
    # Get urls for each synset, without duplicates
    synset_to_urls = {}
    for each_url in tqdm(url_to_synset.keys()):
        if url_to_synset[each_url] not in synset_to_urls:
            synset_to_urls[url_to_synset[each_url]] = []
        synset_to_urls[url_to_synset[each_url]].append(each_url)
    print(len(all_urls) - len(url_to_synset))

    '''
    # Download the asked synsets
    syn_sta = args.syn_sta
    syn_end = min(args.syn_sta + args.syn_len, len(dwnld_synsets))
    for each_synset in dwnld_synsets[syn_sta:syn_end]:
        #now_urls = synset_to_urls[each_synset]
        #print('Now downloading %s, including %i urls' % \
        #        (each_synset, len(now_urls)))
        #download_images_by_synset(each_synset, now_urls, args.add_dir)
        if each_synset in add_synsets:
            continue
        download_tar_by_synset(each_synset, args.add_dir)
    '''

if __name__=='__main__':
    #download_images_by_synset(['n03797390'])
    #download_images_by_synset(['n10091651'])
    main()
