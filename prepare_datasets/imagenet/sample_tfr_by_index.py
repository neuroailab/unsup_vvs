import tensorflow as tf
import numpy as np
import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to generate tfrs according to index')
    parser.add_argument(
            '--save_dir', 
            default='/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label_p01_balanced', 
            type=str, action='store', 
            help='Directory to save the results')
    parser.add_argument(
            '--load_dir', 
            default='/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label_full_widx', 
            type=str, action='store', help='Directory to load the tfrecords')
    parser.add_argument(
            '--img_per_file', 
            default=130, 
            type=int, action='store', 
            help='Number of images per file')
    parser.add_argument(
            '--index_path', 
            default='/mnt/fs1/Dataset/TFRecord_Imagenet_standard/imagenet_p01_balanced_index.npy', 
            type=str, action='store', 
            help='Index that will be used')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    file_num = 1024
    dataformat = 'train-%05i-of-01024'
    train_file_list = [\
            os.path.join(args.load_dir, dataformat % (tmp_indx)) \
            for tmp_indx in range(0, file_num)]
    os.system('mkdir -p %s' % args.save_dir)

    all_index = np.load(args.index_path)
    curr_index = 0
    curr_tfr_idx = 0

    curr_save_tfr_idx = 0
    num_save_tfrs = int(np.ceil(len(all_index) / args.img_per_file))
    save_dataformat = os.path.join(args.save_dir, 'train-%05i-of-%05i')
    output_iter = tf.python_io.TFRecordWriter(
            save_dataformat % (curr_save_tfr_idx, num_save_tfrs))
    curr_save_no_tfr = 0

    while curr_index < np.max(all_index):
        curr_input_file = train_file_list[curr_tfr_idx]
        input_iter = tf.python_io.tf_record_iterator(path=curr_input_file)
        curr_tfr_idx += 1

        example = tf.train.Example()
        for curr_input_rec in input_iter:
            if curr_index in all_index:
                output_iter.write(curr_input_rec)
                curr_save_no_tfr += 1

                if curr_save_no_tfr == args.img_per_file:
                    output_iter.close()
                    if curr_index < np.max(all_index):
                        curr_save_tfr_idx += 1
                        curr_save_no_tfr = 0
                        output_iter = tf.python_io.TFRecordWriter(
                                save_dataformat % (curr_save_tfr_idx, num_save_tfrs))

            example.ParseFromString(curr_input_rec)
            idx_decode = int(example.features.feature['index'].int64_list.value[0])
            assert idx_decode == curr_index, (idx_decode, curr_index)
            curr_index += 1
    os.system('ln -s %s %s' % (os.path.join(args.load_dir, 'validat*'), args.save_dir))


if __name__=="__main__":
    main()
