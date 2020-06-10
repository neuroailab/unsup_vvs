import argparse
import pdb
import torch
import torch.backends.cudnn as cudnn
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
try:
    import cPickle
    pickle = cPickle
except:
    import pickle
import sys


ORIGINAL_SHAPE = [256, 256, 3]
INPUT_SHAPE = [224, 224 ,3]
TFR_PAT = 'tfrecords'


def get_parser():
    parser = argparse.ArgumentParser(
            description='Generate outputs for neural fitting')

    parser.add_argument(
            '--data', type=str, 
            default='/mnt/fs0/datasets/neural_data'\
                    + '/img_split/V4IT/tf_records/images',
            help='path to stimuli')
    parser.add_argument(
            '--model_ckpt', type=str, 
            default='/mnt/fs6/honglinc/trained_models/res18_Lab_cmc/'\
                    + 'checkpoints/checkpoint_epoch190.pth.tar',
            help='path to model')
    parser.add_argument(
            '--batch_size', default=32, type=int,
            help='mini-batch size')
    parser.add_argument(
            '--model', type=str, 
            default='resnet18', choices=['resnet18', 'resnet50', 'resnet101'])
    parser.add_argument(
            '--save_path', type=str,
            default='/mnt/fs4/chengxuz/v4it_temp_results/'\
                    + 'cmc_res18_nf/V4IT_split_0',
            help='path for storing results')
    parser.add_argument(
            '--dataset_type', type=str, default='hvm')
    parser.add_argument(
            '--resv1', action='store_true')
    return parser


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_tfr_files(data_path):
    # Get tfrecord files
    all_tfrs_path = os.listdir(data_path)
    all_tfrs_path = list(filter(lambda x:TFR_PAT in x, all_tfrs_path))
    all_tfrs_path.sort()
    all_tfrs_path = [os.path.join(data_path, each_tfr) \
            for each_tfr in all_tfrs_path]

    return all_tfrs_path


def load_model_class(args):
    import la_cmc_model
    if not args.resv1:
        model_class = la_cmc_model.LACMCModel(args.model, args.model_ckpt)
    else:
        model_class = la_cmc_model.LACMCV1Model(args.model_ckpt)
    return model_class


LAB_MEAN = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
LAB_STD = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
def tolab_normalize(img):
    sys.path.append(os.path.expanduser('~/RotLocalAggregation/'))
    from src.datasets.imagenet import RGB2Lab
    _RGB2Lab = RGB2Lab()
    img = _RGB2Lab(img)
    img -= LAB_MEAN
    img /= LAB_STD
    return img


class NfOutput(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu")
        self.need_to_make_meta = True
        self.model = load_model_class(args)
        self.all_writers = None

    def get_one_image(self, string_record):
        example = tf.train.Example()
        example.ParseFromString(string_record)
        img_string = (example.features.feature['images']
                                      .bytes_list
                                      .value[0])
        img_array = np.fromstring(img_string, dtype=np.float32)
        img_array = img_array.reshape(ORIGINAL_SHAPE)
        img_array *= 255
        img_array = img_array.astype(np.uint8)
        img_array = np.asarray(
                Image.fromarray(img_array).resize(INPUT_SHAPE[:2]))
        img_array = tolab_normalize(img_array)
        img_array = np.transpose(img_array, [2, 0, 1])
        img_array = img_array.astype(np.float32)
        return img_array

    def get_batches(self, all_records):
        all_images = []
        for string_record in all_records:
            all_images.append(self.get_one_image(string_record))
        all_images = np.stack(all_images, axis=0)
        return all_images

    def get_all_images(self, tfr_path):
        record_iterator = tf.python_io.tf_record_iterator(path=tfr_path)
        all_records = list(record_iterator)
        num_imgs = len(all_records)
        all_images = self.get_batches(all_records)
        return num_imgs, all_images

    def _transfer_outputs(self, outputs):
        outputs = [
                np.asarray(output.float().to(self.device)) \
                for output in outputs]
        outputs = [np.transpose(output, [0, 2, 3, 1]) for output in outputs]
        return outputs

    def _get_batch_outputs(self, curr_batch):
        input_var = torch.autograd.Variable(
                torch.from_numpy(curr_batch).cuda())
        model_outputs = self.model.get_all_layer_outputs(input_var)
        return self._transfer_outputs(model_outputs)

    def _make_meta(self, all_outputs):
        args = self.args
        
        for save_key in self.save_keys:
            curr_folder = os.path.join(args.save_path, save_key)
            os.system('mkdir -p %s' % curr_folder)

        for save_key, curr_output in zip(self.save_keys, all_outputs):
            curr_meta = {
                    save_key: {
                        'dtype': tf.string, 
                        'shape': (), 
                        'raw_shape': tuple(curr_output.shape[1:]),
                        'raw_dtype': tf.float32,
                        }
                    }
            meta_path = os.path.join(
                    args.save_path, 
                    save_key, 'meta.pkl')
            pickle.dump(curr_meta, open(meta_path, 'wb'))
        self.need_to_make_meta = False

    def _make_writers(self, tfr_path):
        args = self.args
        
        all_writers = []
        for save_key in self.save_keys:
            write_path = os.path.join(
                    args.save_path, save_key,
                    os.path.basename(tfr_path))
            writer = tf.python_io.TFRecordWriter(write_path)
            all_writers.append(writer)
        self.all_writers = all_writers

    def _write_outputs(self, all_outputs):
        for writer, curr_output, save_key in \
                zip(self.all_writers, 
                    all_outputs, self.save_keys):
            for idx in range(curr_output.shape[0]):
                curr_value = curr_output[idx]
                save_feature = {
                        save_key: _bytes_feature(curr_value.tostring())
                        }
                example = tf.train.Example(
                        features=tf.train.Features(feature=save_feature))
                writer.write(example.SerializeToString())

    def _close_writers(self):
        for each_writer in self.all_writers:
            each_writer.close()
        self.all_writers = None

    def write_outputs_for_one_tfr(self, tfr_path):
        args = self.args
        if args.dataset_type == 'v1_tc':
            global ORIGINAL_SHAPE
            global INPUT_SHAPE
            ORIGINAL_SHAPE = [80, 80, 3]
            INPUT_SHAPE = [40, 40, 3]
        num_imgs, all_images = self.get_all_images(tfr_path)

        for start_idx in range(0, num_imgs, args.batch_size):
            curr_batch = all_images[start_idx : start_idx + args.batch_size]
            all_outputs = self._get_batch_outputs(curr_batch)

            if self.need_to_make_meta:
                self.save_keys = [
                        'conv%i' % idx for idx in range(len(all_outputs))]
                self._make_meta(all_outputs)
            if self.all_writers is None:
                self._make_writers(tfr_path)
            self._write_outputs(all_outputs)
        self._close_writers()


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.dataset_type == 'v1_tc':
        global TFR_PAT
        TFR_PAT = 'split'
    all_tfr_path = get_tfr_files(args.data)

    nf_output = NfOutput(args)
    
    for tfr_path in tqdm(all_tfr_path):
        nf_output.write_outputs_for_one_tfr(tfr_path)


if __name__ == '__main__':
    main()
