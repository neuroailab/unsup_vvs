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

    # ThreedWorld, not used now
    DATA_PATH['threed/train/images'] = '/mnt/fs0/datasets/one_world_dataset/tfdata/images'
    DATA_PATH['threed/train/normals'] = '/mnt/fs0/datasets/one_world_dataset/tfdata/normals'
    DATA_PATH['threed/val/images'] = '/mnt/fs0/datasets/one_world_dataset/tfvaldata/images'
    DATA_PATH['threed/val/normals'] = '/mnt/fs0/datasets/one_world_dataset/tfvaldata/normals'

    # Scenenet, set which_scenenet to be 2
    DATA_PATH['scenenet/train/images'] = '/mnt/fs1/Dataset/scenenet_combine/photo'
    DATA_PATH['scenenet/train/normals'] = '/mnt/fs1/Dataset/scenenet_combine/normal_new'
    DATA_PATH['scenenet/train/depths'] = '/mnt/fs1/Dataset/scenenet_combine/depth_new'
    #DATA_PATH['scenenet/val/images'] = '/mnt/fs1/Dataset/scenenet_combine_val/photo'
    #DATA_PATH['scenenet/val/normals'] = '/mnt/fs1/Dataset/scenenet_combine_val/normal_new'
    #DATA_PATH['scenenet/val/depths'] = '/mnt/fs1/Dataset/scenenet_combine_val/depth_new'
    DATA_PATH['scenenet/val/images'] = '/mnt/fs1/Dataset/scenenet_combine/photo'
    DATA_PATH['scenenet/val/normals'] = '/mnt/fs1/Dataset/scenenet_combine/normal_new'
    DATA_PATH['scenenet/val/depths'] = '/mnt/fs1/Dataset/scenenet_combine/depth_new'

    DATA_PATH['scenenet_compress/train/images'] = '/mnt/fs0/chengxuz/Data/scenenet_compress/photo'
    DATA_PATH['scenenet_compress/train/normals'] = '/mnt/fs0/chengxuz/Data/scenenet_compress/normal_new'
    DATA_PATH['scenenet_compress/train/depths'] = '/mnt/fs0/chengxuz/Data/scenenet_compress/depth_new'
    DATA_PATH['scenenet_compress/val/images'] = '/mnt/fs0/chengxuz/Data/scenenet_compress_val/photo'
    DATA_PATH['scenenet_compress/val/normals'] = '/mnt/fs0/chengxuz/Data/scenenet_compress_val/normal_new'
    DATA_PATH['scenenet_compress/val/depths'] = '/mnt/fs0/chengxuz/Data/scenenet_compress_val/depth_new'

    #scenenet_new_prefix = '/mnt/fs0/chengxuz/Data'
    scenenet_new_prefix = '/mnt/fs1/Dataset/scenenet_all'

    if host=='kanefsky':
        scenenet_new_prefix = '/mnt/data3/chengxuz/Dataset'
    elif host=='icst2' or host=='icst3' or host=='icst4' or host=='icst5' or host=='icst6':
        scenenet_new_prefix = '/S1/LCWM/harry/Dataset'

    if not overall_local is None:
        scenenet_new_prefix = '%s/scenenet_all' % overall_local

    # Saved in png and use new normal computing method
    DATA_PATH['scenenet_new/train/images'] = '%s/scenenet_new/photo' % scenenet_new_prefix
    DATA_PATH['scenenet_new/train/normals'] = '%s/scenenet_new/normals' % scenenet_new_prefix
    DATA_PATH['scenenet_new/train/depths'] = '%s/scenenet_new/depth' % scenenet_new_prefix
    #DATA_PATH['scenenet_new/train/instances'] = '%s/scenenet_new/instance' % scenenet_new_prefix
    DATA_PATH['scenenet_new/train/instances'] = '%s/scenenet_new/classes' % scenenet_new_prefix
    DATA_PATH['scenenet_new/val/images'] = '%s/scenenet_new_val/photo' % scenenet_new_prefix
    DATA_PATH['scenenet_new/val/normals'] = '%s/scenenet_new_val/normals' % scenenet_new_prefix
    DATA_PATH['scenenet_new/val/depths'] = '%s/scenenet_new_val/depth' % scenenet_new_prefix
    #DATA_PATH['scenenet_new/val/instances'] = '%s/scenenet_new_val/instance' % scenenet_new_prefix
    DATA_PATH['scenenet_new/val/instances'] = '%s/scenenet_new_val/classes' % scenenet_new_prefix
    DATA_PATH['scenenet_new/depth_mlt'] = '%s/scenenet_new/depth_mlt' % scenenet_new_prefix

    #pbrnet_prefix = '/mnt/fs0/chengxuz/Data/pbrnet_folder'
    pbrnet_prefix = '/mnt/fs1/Dataset/pbrnet'
    if host=='kanefsky':
        pbrnet_prefix = '/mnt/data3/chengxuz/Dataset/pbrnet'
    elif host=='icst2' or host=='icst3' or host=='icst4' or host=='icst5' or host=='icst6':
        pbrnet_prefix = '/S1/LCWM/harry/Dataset/pbrnet'

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

    #imagenet_prefix = '/mnt/fs0/datasets'
    imagenet_prefix = '/mnt/fs1/Dataset'
    if host=='kanefsky':
        imagenet_prefix = '/mnt/data'
    elif host=='icst2' or host=='icst3' or host=='icst4' or host=='icst5' or host=='icst6':
        imagenet_prefix = '/S1/LCWM/harry'
        print("***********Get into the icst server************")

    if not localimagenet==None:
        imagenet_prefix = localimagenet

    if not overall_local is None:
        imagenet_prefix = overall_local

    # ImageNet
    DATA_PATH['imagenet/image_label_full'] \
            = '%s/TFRecord_Imagenet_standard/image_label_full' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part'] \
            = '%s/TFRecord_Imagenet_standard/image_label_full_part' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part1_balanced'] \
            = '%s/TFRecord_Imagenet_standard/image_label_p01_balanced' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part2_balanced'] \
            = '%s/TFRecord_Imagenet_standard/image_label_p02_balanced' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part3'] \
            = '%s/TFRecord_Imagenet_standard/imagenet_p03_ub' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part3_widx_ordrd'] \
            = '%s/TFRecord_Imagenet_standard/image_label_p03_widx_ordered' % imagenet_prefix
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
    DATA_PATH['imagenet/image_label_full_infant'] \
            = '%s/new_imagenet_tfr' % imagenet_prefix
    DATA_PATH['imagenet/image_label_full_infant_ctl_es'] \
            = '%s/new_imagenet_ctl_eq_smpl_tfr' % imagenet_prefix
    DATA_PATH['imagenet/image_label_full_infant_ctl'] \
            = '%s/new_imagenet_ctl_tfr' % imagenet_prefix
    DATA_PATH['imagenet/image_label_full_infant_widx'] \
            = '%s/new_imagenet_tfr_widx' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part30_widx_ordrd'] \
            = '%s/TFRecord_Imagenet_standard/image_label_p30_widx_ordered' % imagenet_prefix
    DATA_PATH['imagenet/image_label_part70_widx_ordrd'] \
            = '%s/TFRecord_Imagenet_standard/image_label_p70_widx_ordered' % imagenet_prefix

    # Coco, use no 0 for new tensorflow
    #coco_prefix = '/mnt/fs0/datasets/mscoco'
    coco_prefix = '/mnt/fs1/Dataset/mscoco'
    if host=='kanefsky':
        coco_prefix = '/mnt/data3/chengxuz/Dataset/coco_dataset'

    if not overall_local is None:
        coco_prefix = '%s/mscoco' % overall_local

    FOLDERs = { 'train': '%s/train_tfrecords' % coco_prefix,
                'val':  '%s/val_tfrecords' % coco_prefix}
    KEY_LIST = ['bboxes', 'height', 'images', 'labels', 'num_objects', \
            'segmentation_masks', 'width']
    for key_group in FOLDERs:
        for key_feature in KEY_LIST:
            DATA_PATH[ 'coco/%s/%s' % (key_group, key_feature) ] = os.path.join(FOLDERs[key_group], key_feature)

    FOLDERs_no0 = { 'train': '%s/train_tfrecords_no0' % coco_prefix,
                'val':  '%s/val_tfrecords_no0' % coco_prefix}
    for key_group in FOLDERs_no0:
        for key_feature in KEY_LIST:
            DATA_PATH[ 'coco_no0/%s/%s' % (key_group, key_feature) ] = os.path.join(FOLDERs_no0[key_group], key_feature)

    # Places
    #place_prefix = '/mnt/fs0/chengxuz/Data'
    place_prefix = '/mnt/fs1/Dataset'
    if not overall_local is None:
        place_prefix = overall_local

    DATA_PATH['place/train/images'] = '%s/places/tfrs_train/image' % place_prefix
    DATA_PATH['place/train/labels'] = '%s/places/tfrs_train/label' % place_prefix
    DATA_PATH['place/train/images_part'] = '%s/places/tfrs_train/image_part' % place_prefix
    DATA_PATH['place/train/labels_part'] = '%s/places/tfrs_train/label_part' % place_prefix
    DATA_PATH['place/val/images'] = '%s/places/tfrs_val/image' % place_prefix
    DATA_PATH['place/val/labels'] = '%s/places/tfrs_val/label' % place_prefix
    DATA_PATH['place/val/images_part'] = '%s/places/tfrs_val/image' % place_prefix
    DATA_PATH['place/val/labels_part'] = '%s/places/tfrs_val/label' % place_prefix

    # Nyuv2, only for validation
    DATA_PATH['nyuv2/val/images'] = '/mnt/fs0/chengxuz/Data/nyuv2/labeled/image'
    DATA_PATH['nyuv2/val/depths'] = '/mnt/fs0/chengxuz/Data/nyuv2/labeled/depth'

    # Kinetics
    kinetics_prefix = '/mnt/fs1/Dataset/kinetics/'
    #ki_FOLDERs = { 'train': '%s/train_tfrs' % kinetics_prefix,
    #            'val':  '%s/val_tfrs' % kinetics_prefix}
    ki_FOLDERs = { 'train': '%s/train_tfrs_5fps' % kinetics_prefix,
                'val':  '%s/val_tfrs_5fps' % kinetics_prefix}
    ki_KEY_LIST = ['path', 'label_p']
    for key_group in ki_FOLDERs:
        for key_feature in ki_KEY_LIST:
            DATA_PATH[ 'kinetics/%s/%s' % (key_group, key_feature) ] = os.path.join(ki_FOLDERs[key_group], key_feature)

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
