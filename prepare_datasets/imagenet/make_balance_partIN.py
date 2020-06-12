import numpy as np
import pickle
import argparse

NUM_CAT = 1000


def get_config():
    parser = argparse.ArgumentParser(
            description='Get balanced labeled part from all_label.npy')
    parser.add_argument(
            '--lbl_save_path', 
            default=None, type=str, required=True,
            action='store', help='Path to save the label')
    parser.add_argument(
            '--idx_save_path', 
            default=None, type=str, required=True,
            action='store', help='Path to save the index')
    parser.add_argument(
            '--lbl_pkl_path', 
            default=None, type=str, required=True,
            action='store', help='Path where label pkl is saved')
    parser.add_argument(
            '--per_cat_img', 
            default=None, type=int, required=True, action='store', 
            help='Number of labeled images per category')
    return parser


def main():
    cfg = get_config()
    args = cfg.parse_args()

    part_label = []
    part_index = []
    curr_idx = 0
    no_img_cat = {cat_idx: 0 for cat_idx in range(NUM_CAT)}

    all_labels = pickle.load(open(args.lbl_pkl_path, 'rb'))

    while len(part_label) < args.per_cat_img * NUM_CAT:
        curr_label = all_labels[curr_idx]
        if no_img_cat[curr_label] < args.per_cat_img:
            no_img_cat[curr_label] += 1
            part_label.append(curr_label)
            part_index.append(curr_idx)
        curr_idx += 1

    np.save(args.lbl_save_path, np.asarray(part_label))
    np.save(args.idx_save_path, np.asarray(part_index))


if __name__ == '__main__':
    main()
