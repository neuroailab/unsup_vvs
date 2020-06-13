import os, sys
host = os.uname()[1]


def which_imagenet_map(whichimagenet):
    if not whichimagenet.isdigit():
        return whichimagenet
    whichimagenet = int(whichimagenet)
    # mapping function for previous commands with number whichimagenet 
    mapping_dict = {
            5: 'full',
            6: 'part',
            9: 'part5',
            10: 'part1',
            11: 'part3',
            12: 'part1_b',
            13: 'part2',
            20: 'full_widx',
            40: 'full_new',
            }
    if whichimagenet in mapping_dict:
        return mapping_dict[whichimagenet]
    else:
        return whichimagenet


def get_data_path(localimagenet=None, overall_local=None):
    # Pathes for all the datasets
    DATA_PATH = {}

    pbrnet_prefix = '/mnt/fs1/Dataset/pbrnet'
    if not overall_local is None:
        pbrnet_prefix = '%s/pbrnet' % overall_local

    # Pbrnet
    DATA_PATH['pbrnet/train/images'] = '%s/tfrecords/mlt' % pbrnet_prefix
    DATA_PATH['pbrnet/train/normals'] = '%s/tfrecords/normal' % pbrnet_prefix
    DATA_PATH['pbrnet/train/depths'] = '%s/tfrecords/depth' % pbrnet_prefix
    DATA_PATH['pbrnet/train/instances'] = '%s/tfrecords/category' % pbrnet_prefix
    DATA_PATH['pbrnet/val/images'] = '%s/tfrecords_val/mlt' % pbrnet_prefix
    DATA_PATH['pbrnet/val/normals'] = '%s/tfrecords_val/normal' % pbrnet_prefix
    DATA_PATH['pbrnet/val/depths'] = '%s/tfrecords_val/depth' % pbrnet_prefix
    DATA_PATH['pbrnet/val/instances'] = '%s/tfrecords_val/category' % pbrnet_prefix
    DATA_PATH['pbrnet/depth_mlt'] = '%s/tfrecords/depth_mlt' % pbrnet_prefix

    imagenet_prefix = '/mnt/fs1/Dataset'
    if not localimagenet==None:
        imagenet_prefix = localimagenet
    if not overall_local is None:
        imagenet_prefix = overall_local

    # ImageNet
    DATA_PATH['imagenet/image_label_full'] \
            = '%s/TFRecord_Imagenet_standard/image_label_full_widx' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part1_balanced'] \
            = '%s/TFRecord_Imagenet_standard/image_label_p01_balanced' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part2_balanced'] \
            = '%s/TFRecord_Imagenet_standard/image_label_p02_balanced' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part3'] \
            = '%s/TFRecord_Imagenet_standard/imagenet_p03_ub' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part4_balanced'] \
            = '%s/TFRecord_Imagenet_standard/image_label_p04_balanced' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part5'] \
            = '%s/TFRecord_Imagenet_standard/image_label_part5' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part6_balanced'] \
            = '%s/TFRecord_Imagenet_standard/image_label_p06_balanced' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part10'] \
            = '%s/TFRecord_Imagenet_standard/image_label_part10' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part20_balanced'] \
            = '%s/TFRecord_Imagenet_standard/image_label_p20_balanced' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part50_balanced'] \
            = '%s/TFRecord_Imagenet_standard/image_label_p50_balanced' % imagenet_prefix
    DATA_PATH['imagenet/image_label_full_widx'] \
            = '%s/TFRecord_Imagenet_standard/image_label_full_widx' % imagenet_prefix

    DATA_PATH['saycam/frames'] = '/mnt/fs4/chengxuz/Dataset/saycam_frames_tfr'
    DATA_PATH['saycam/all_frames'] = '/mnt/fs4/chengxuz/Dataset/saycam_all_frames_tfr'
    return DATA_PATH


def get_TPU_data_path():
    data_path = {}

    data_path['imagenet/image_label_full'] = \
            'gs://full_size_imagenet/image_label_full/'
    data_path['imagenet/image_label_full_widx'] = \
            'gs://full_size_imagenet/image_label_full_widx/'
    #data_path['imagenet/image_label_part'] = 'gs://small-imagenet/'
    data_path['imagenet/image_label_part'] = \
            'gs://small-imagenet-random/image_label_full_part/'
    data_path['imagenet/image_label_part10'] = \
            'gs://small-imagenet-random/image_label_part10/'
    data_path['imagenet/image_label_part5'] = 'gs://smaller-imagenet/'
    data_path['imagenet/image_label_part1'] = \
            'gs://small-imagenet-p01/imagenet_p01_ub/'
    data_path['imagenet/image_label_part2'] = \
            'gs://small-imagenet-p02/imagenet_p02_ub/'
    data_path['imagenet/image_label_part3'] = \
            'gs://small-imagenet-p03/imagenet_p03_ub/'
    data_path['imagenet/image_label_part1_b'] = \
            'gs://small-imagenet-p01/imagenet_p01/'
    data_path['imagenet/image_label_infant'] = \
            'gs://infant_imagenet/new_imagenet_tfr/'
    data_path['imagenet/image_label_infant_ctl'] = \
            'gs://infant_imagenet/new_imagenet_ctl_tfr/'
    data_path['imagenet/image_label_infant_ctl_es'] = \
            'gs://infant_imagenet/new_imagenet_ctl_eq_smpl_tfr/'
    data_path['imagenet/image_label_infant_02'] = \
            'gs://infant_imagenet/infant_imagenet_02/'
    data_path['imagenet/image_label_infant_03'] = \
            'gs://infant_imagenet/infant_imagenet_03/'
    data_path['imagenet/image_label_infant_05'] = \
            'gs://infant_imagenet/infant_imagenet_05/'
    data_path['imagenet/image_label_infant_10'] = \
            'gs://infant_imagenet/infant_imagenet_10/'
    data_path['imagenet/image_label_infant_20'] = \
            'gs://infant_imagenet/infant_imagenet_20/'

    return data_path
