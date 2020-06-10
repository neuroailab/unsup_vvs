"""Input pipeline using tf.data.Dataset."""
import os
import copy
import tensorflow.compat.v1 as tf
import pdb
from tqdm import tqdm


# Useful util function
def fetch_dataset(filename):
    buffer_size = 8 * 1024 * 1024     # 8 MiB per file
    dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
    return dataset


class SAYCam(object):
    def __init__(
            self,
            dataset,
            data_dir,
            *args, **kwargs):
        assert dataset == 'SAYCam', \
                "Must be SAYCam dataset"
        self.dataset = dataset
        self.data_dir = data_dir
        self.num_examples = 1281088
        self.num_classes = 1000

    def get_tfr_filenames(self, folder_name, file_pattern='*.tfrecords'):
        # Get list of tfrecord filenames for given folder_name 
        # fitting the given file_pattern
        tfrecord_pattern = os.path.join(folder_name, file_pattern)
        datasource = tf.gfile.Glob(tfrecord_pattern)
        datasource.sort()
        return datasource

    def _get_imgnt_list_dataset(self, curr_data_dir):
        file_pattern = 'train-*' if self.is_training else 'validation-*'
        list_file = self.get_tfr_filenames(curr_data_dir, file_pattern)
        file_bf_size = len(list_file)
        list_file_dataset = tf.data.Dataset.list_files(list_file)
        return list_file_dataset, file_bf_size

    def __parse_ImageNet(self, value):
        # Parse the tfrecord
        keys_to_features = {
                'images': tf.FixedLenFeature((), tf.string, ''),
                'labels': tf.FixedLenFeature([], tf.int64, -1)
                }
        parsed = tf.parse_single_example(value, keys_to_features)
        img = tf.image.decode_image(
                parsed['images'], channels=3, dtype=tf.uint8)
        img.set_shape((None, None, 3))
        img = img[::-1, :, :]
        return img, parsed['labels']

    def as_dataset(self, split, shuffle_files, as_supervised):
        assert split == 'train', "Must be in training phase"
        self.is_training = True
        list_file_dataset, file_bf_size \
                = self._get_imgnt_list_dataset(self.data_dir)
        # Function to fetch and parse
        def _fetch_and_parse(dataset):
            # Shuffle the file list dataset if needed
            dataset = dataset.apply(
                    tf.data.experimental.shuffle_and_repeat(
                        file_bf_size))
            # Fetch the tfrecords
            dataset = dataset.apply(
                    tf.data.experimental.parallel_interleave(
                        fetch_dataset, 
                        cycle_length=8, sloppy=True))
            dataset = dataset.map(
                    self.__parse_ImageNet,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            return dataset

        # Repeat the file list if needed
        dataset = _fetch_and_parse(list_file_dataset)
        return dataset


def saycam_debug():
    from PIL import Image
    builder = SAYCam('SAYCam', 'gs://full_size_imagenet/saycam_frames_tfr/')
    dataset = builder.as_dataset('train', True, True)
    data_iter = tf.data.make_one_shot_iterator(dataset).get_next()
    sess = tf.Session()
    out_folder = '/home/chengxuz/saycam_tf_out_imgs'
    os.system('mkdir -p ' + out_folder)
    for _idx in tqdm(range(100)):
        data_val = sess.run(data_iter)
        _img = Image.fromarray(data_val[0])
        _img.save(os.path.join(out_folder, 'img%i.png' % _idx), 'PNG')


def in_debug():
    import tensorflow_datasets as tfds
    builder = tfds.builder('imagenet2012', data_dir='gs://full_size_imagenet/tfrs/')
    dataset = builder.as_dataset(split='train', shuffle_files=True, as_supervised=True)
    data_iter = tf.data.make_one_shot_iterator(dataset).get_next()
    sess = tf.Session()
    data_val = sess.run(data_iter)
    pdb.set_trace()
    pass


if __name__ == '__main__':
    saycam_debug()
    #in_debug()
