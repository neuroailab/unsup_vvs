import sys, os
import numpy as np
import argparse
import pdb

def process_one_month(curr_month):
    curr_month = curr_month.replace('t', '')
    if curr_month=='*':
        curr_month = 0
    else:
        curr_month = int(curr_month)
    return curr_month

def get_one_noun(curr_split):
    offset_len = 0
    # The first is the name
    name_len = 0
    while curr_split[name_len].isalpha():
        name_len += 1
    ## Concatenate the name strings
    name = curr_split[0]
    for each_name in curr_split[1:name_len]:
        name = name + ' ' + each_name
    offset_len += name_len - 1
    # For comprehensive month
    compre_pos = 1
    first_big = 0 
    if curr_split[compre_pos + offset_len]=='>':
        first_big = 1
    offset_len += first_big
    compre_month = curr_split[compre_pos + offset_len]
    compre_month = process_one_month(compre_month)
    # For produce month
    produce_pos = 3
    second_big = 0
    if curr_split[produce_pos + offset_len]=='>':
        second_big = 1
    offset_len += second_big
    produce_month = curr_split[produce_pos + offset_len]
    produce_month = process_one_month(produce_month)
    # Cut the first object related strings
    cut_len = 5 + offset_len
    return [name, compre_month, produce_month, first_big, second_big], \
           curr_split[cut_len:]

def load_noun_dict():
    ret_dict = {}
    file_path = 'infant_nouns_raw.txt'
    fin = open(file_path, 'r')
    all_lines = fin.readlines()
    for each_line in all_lines:
        each_line = each_line.replace('.', '')
        each_line = each_line.replace('\'s', '')
        while '  ' in each_line:
            each_line = each_line.replace('  ', ' ')
        now_line = each_line[:-2]
        now_split = now_line.split(' ')
        while len(now_split)>0:
            now_noun, now_split = get_one_noun(now_split)
            ret_dict[now_noun[0]] = {
                    'compre_month':now_noun[1],
                    'produce_month':now_noun[2],
                    'first_big':now_noun[3],
                    'second_big':now_noun[4],
                    }
    return ret_dict

if __name__=='__main__':
    noun_dict = load_noun_dict()
    print(len(noun_dict.keys()))
    non_zero_keys = [each_key for each_key in noun_dict \
            if noun_dict[each_key]['compre_month']>0 and noun_dict[each_key]['first_big']==0]
    print(len(non_zero_keys))
    print(non_zero_keys)
    non_zero_fs_keys = [each_key for each_key in noun_dict \
            if noun_dict[each_key]['second_big']==0]
    print(len(non_zero_fs_keys))
    print(non_zero_fs_keys)
