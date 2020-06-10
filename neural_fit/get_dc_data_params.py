import numpy as np
import os
import pdb


def get_vgg16_params(data_path, shape_dict):
    data_path.pop('images')
    shape_dict.pop('images')

    dc_home = '/data5/chengxuz/Dataset/v4it_temp_results/V4IT'
    data_path['conv13'] = dc_home + '/conv13'
    shape_dict['conv13'] = (14, 14, 512)

    data_path['conv12'] = dc_home + '/conv12'
    shape_dict['conv12'] = (14, 14, 512)

    data_path['conv11'] = dc_home + '/conv11'
    shape_dict['conv11'] = (14, 14, 512)

    data_path['conv10'] = dc_home + '/conv10'
    shape_dict['conv10'] = (28, 28, 512)

    data_path['conv9'] = dc_home + '/conv9'
    shape_dict['conv9'] = (28, 28, 512)

    data_path['conv7'] = dc_home + '/conv7'
    shape_dict['conv7'] = (56, 56, 256)
    return data_path, shape_dict


def get_res18_params(data_path, shape_dict):
    data_path.pop('images')
    shape_dict.pop('images')

    dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/deepcluster_hvm/res18/V4IT'
    data_path['conv105'] = dc_home + '/conv105'
    shape_dict['conv105'] = (56, 56, 64)

    data_path['conv106'] = dc_home + '/conv106'
    shape_dict['conv106'] = (28, 28, 128)

    data_path['conv107'] = dc_home + '/conv107'
    shape_dict['conv107'] = (14, 14, 256)

    data_path['conv108'] = dc_home + '/conv108'
    shape_dict['conv108'] = (7, 7, 512)
    return data_path, shape_dict


def get_res18_deseq_params(data_path, shape_dict, args):
    data_path.pop('images')
    shape_dict.pop('images')

    dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/deepcluster_hvm/res18_np/V4IT_%s'
    filter_list = [64, 64, 64, 128, 128, 256, 256, 512, 512]
    shape_list = [56, 56, 56, 28, 28, 14, 14, 7, 7]
    if args.deepcluster == 'res18_deseq_v1':
        dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/deepcluster_hvm/res18_np/V1_%s'
        shape_list = [10, 10, 10, 5, 5, 3, 3, 2, 2]

    dc_home = dc_home % args.which_split
    for layer_number in range(104, 113):
        layer_name = 'conv%i' % layer_number
        data_path[layer_name] = dc_home + '/' + layer_name
        shape_now = shape_list[layer_number - 104]
        filter_now = filter_list[layer_number - 104]
        shape_dict[layer_name] = (shape_now, shape_now, filter_now)
    return data_path, shape_dict


def get_res_all_data_shape(
        host_dir, args, layer_offset, v1_shape,
        filter_types = [64, 256, 512, 1024, 2048],
        type_nums = [1, 3, 4, 6, 3]):
    shape_types = [56, 56, 28, 14, 7]
    if v1_shape:
        shape_types = [10, 10, 5, 3, 2]
    filter_list = []
    shape_list = []
    for _filter, _shape, _num in zip(filter_types, shape_types, type_nums):
        filter_list.extend([_filter] * _num)
        shape_list.extend([_shape] * _num)

    data_path = {}
    shape_dict = {}
    for _idx in range(np.sum(type_nums)):
        layer_name = 'conv%i' % (_idx + layer_offset)
        data_path[layer_name] = os.path.join(host_dir, layer_name)
        shape_now = shape_list[_idx]
        filter_now = filter_list[_idx]
        shape_dict[layer_name] = (shape_now, shape_now, filter_now)
    return data_path, shape_dict


def update_data_shape_dict(
        data_path, shape_dict, args, 
        all_data_path, all_shape_dict):
    data_path.pop('images')
    shape_dict.pop('images')

    all_layers = []
    if args.it_nodes is not None:
        all_layers.extend(args.it_nodes.split(','))
    if args.v4_nodes is not None:
        all_layers.extend(args.v4_nodes.split(','))

    for layer_name in all_layers:
        data_path[layer_name] = all_data_path[layer_name]
        shape_dict[layer_name] = all_shape_dict[layer_name]
    return data_path, shape_dict


def get_res50_deseq_params(data_path, shape_dict, args):
    dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/deepcluster_hvm/res50/V4IT_%s'
    v1_shape = False
    if args.deepcluster == 'res50_deseq_v1':
        dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/deepcluster_hvm/res50/V1_%s'
        v1_shape = True 

    dc_home = dc_home % args.which_split
    all_data_path, all_shape_dict = get_res_all_data_shape(
            dc_home, args, 104, v1_shape)
    return update_data_shape_dict(
            data_path, shape_dict, args, 
            all_data_path, all_shape_dict)


def get_cmc_res50_params(data_path, shape_dict, args):
    which_part = args.deepcluster.split(':')[1]
    dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/cmc_nf/res50/V4IT_%s/%s' \
            % (args.which_split, which_part)
    v1_shape = False
    if args.deepcluster.split(':')[0] == 'cmc_res50_v1':
        dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/cmc_nf/res50/V1_%s/%s' \
                % (args.which_split, which_part)
        v1_shape = True 

    all_data_path, all_shape_dict = get_res_all_data_shape(
            dc_home, args, 0, v1_shape)
    return update_data_shape_dict(
            data_path, shape_dict, args, 
            all_data_path, all_shape_dict)


def get_cmc_res18_params(data_path, shape_dict, args):
    dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/cmc_res18_nf/V4IT_%s' \
            % args.which_split
    v1_shape = False
    if args.deepcluster == 'cmc_res18_v1':
        dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/cmc_res18_nf/V1_%s' \
                % args.which_split
        v1_shape = True 

    all_data_path, all_shape_dict = get_res_all_data_shape(
            dc_home, args, 0, v1_shape,
            filter_types=[64, 64, 128, 256, 512],
            type_nums=[1, 2, 2, 2, 2])
    return update_data_shape_dict(
            data_path, shape_dict, args, 
            all_data_path, all_shape_dict)


def get_la_cmc_res18_params(data_path, shape_dict, args):
    dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/la_cmc_res18_nf/V4IT_%s' \
            % args.which_split
    v1_shape = False
    if args.deepcluster == 'la_cmc_res18_v1':
        dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/la_cmc_res18_nf/V1_%s' \
                % args.which_split
        v1_shape = True 

    all_data_path, all_shape_dict = get_res_all_data_shape(
            dc_home, args, 0, v1_shape,
            filter_types=[64, 64, 128, 256, 512],
            type_nums=[1, 2, 2, 2, 2])
    return update_data_shape_dict(
            data_path, shape_dict, args, 
            all_data_path, all_shape_dict)


def get_la_cmc_res18v1_params(data_path, shape_dict, args):
    dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/la_cmc_res18v1_nf/V4IT_%s' \
            % args.which_split
    v1_shape = False
    if args.deepcluster == 'la_cmc_res18v1_v1':
        dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/la_cmc_res18v1_nf/V1_%s' \
                % args.which_split
        v1_shape = True 

    all_data_path, all_shape_dict = get_res_all_data_shape(
            dc_home, args, 0, v1_shape,
            filter_types=[64, 64, 128, 256, 512],
            type_nums=[1, 2, 2, 2, 2])
    return update_data_shape_dict(
            data_path, shape_dict, args, 
            all_data_path, all_shape_dict)


def get_pt_official_res18_params(data_path, shape_dict, args):
    dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/pt_official_res18_nf/V4IT_%s' \
            % args.which_split
    v1_shape = False
    if args.deepcluster == 'pt_official_res18_v1':
        dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/pt_official_res18_nf/V1_%s' \
                % args.which_split
        v1_shape = True 
    if args.deepcluster == 'pt_official_res18_var6':
        dc_home = '/mnt/fs4/chengxuz/v4it_temp_results/pt_official_res18_nf/hvm_var6_%s' \
                % args.which_split

    all_data_path, all_shape_dict = get_res_all_data_shape(
            dc_home, args, 0, v1_shape,
            filter_types=[64, 64, 128, 256, 512],
            type_nums=[1, 2, 2, 2, 2])
    return update_data_shape_dict(
            data_path, shape_dict, args, 
            all_data_path, all_shape_dict)
