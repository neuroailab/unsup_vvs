import tensorflow as tf
import argparse
import os
import numpy as np
import sys
import dldata.stimulus_sets.hvm as hvm

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_parser():
    parser = argparse.ArgumentParser(description='The script to write to tfrecords combining images and labels')
    #parser.add_argument('--savekey', default = 'labels_ave', type = str, action = 'store', help = 'Key to store the averaged responses')
    parser.add_argument('--dir', default = '/mnt/fs0/datasets/neural_data/img_split/V4IT', type = str, action = 'store', help = 'Directory to load the tfrecords')
    parser.add_argument('--savedir', default = '/mnt/fs1/Dataset/neural_resp/V4IT', type = str, action = 'store', help = 'Directory to save the tfrecords')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    dataset_hvm = hvm.HvMWithDiscfade()
    neural_fea  = dataset_hvm.neuronal_features

    IT_neurons = dataset_hvm.IT_NEURONS
    V4_neurons = dataset_hvm.V4_NEURONS

    npz_path = os.path.join(args.dir, 'meta.npz')

    all_indices = np.load(npz_path)
    train_indices = all_indices['arr_0'][()]['train_indices']
    test_indices = all_indices['arr_0'][()]['test_indices']

    key_dict = {'IT_ave': IT_neurons, 'V4_ave': V4_neurons}
    key_split = {'train': train_indices, 'test': test_indices}

    nm_per_tf = 256

    #train_labels = neural_fea[train_indices]
    #print(neural_fea[train_indices[0]])
    #print(train_labels[0])
    #print(train_indices[0])

    # Write into each split and each group
    for which_group in key_dict:
        curr_neuron_indices = key_dict[which_group]

        # Creat current split
        curr_dir = os.path.join(args.savedir, which_group)
        os.system('mkdir -p %s' % curr_dir)

        for which_split in key_split:
            curr_image_indices = key_split[which_split]
            #print(len(curr_image_indices))
            #print(curr_neuron_indices)
            #print(neural_fea[curr_image_indices].shape)

            curr_responses = neural_fea[curr_image_indices][:, curr_neuron_indices]
            print(curr_responses.shape)
            len_images = len(curr_image_indices)

            curr_file_indx = 0
            for image_start_indx in xrange(0, len_images, nm_per_tf):
                tfrecord_name = os.path.join(curr_dir, '%s_%i-%s.tfrecords' % (which_split, curr_file_indx, which_group))
                writer = tf.python_io.TFRecordWriter(tfrecord_name)
                #print(tfrecord_name)

                for curr_indx in xrange(image_start_indx, image_start_indx + nm_per_tf):
                    curr_arr = curr_responses[curr_indx]
                    curr_arr = curr_arr.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                              which_group: _bytes_feature(curr_arr)}))
                    writer.write(example.SerializeToString())

                curr_file_indx = curr_file_indx + 1
                writer.close()

if __name__=="__main__":
    main()
