import models.network_cfg_scripts.shared_funcs as shared_funcs
import copy
import pdb


def get_rp_resnet_18_encode_head(input_layer):
    ret_cfg = {"input": input_layer}
    for layer_idx in range(1, 3):
        ret_cfg[layer_idx] = shared_funcs.get_resnet_basicblock_one_layer(128)
    return ret_cfg


def get_rp_resnet_18_encode_head_bg(input_layer):
    ret_cfg = {"input": input_layer}
    ret_cfg[1] = {
            "ResBlock": [
                {"filter_size": 1, "stride": 1, "num_filters": 128, "bn": 1}, 
                {"filter_size": 3, "stride": 1, "num_filters": 128, "bn": 1}, 
                {"filter_size": 1, "stride": 1, "num_filters": 1024, "bn": 1}]
            }
    ret_cfg[2] = {
            "ResBlock": [
                {"filter_size": 1, "stride": 1, "num_filters": 128, "bn": 1}, 
                {"filter_size": 3, "stride": 1, "num_filters": 128, "bn": 1}, 
                {"filter_size": 1, "stride": 1, "num_filters": 512, "bn": 1}]
            }
    return ret_cfg


def get_rp_resnet_18_rp_category():
    ret_cfg = {"as_output": 1}
    for layer_idx in range(1, 4):
        ret_cfg[layer_idx] = shared_funcs.get_resnet_basicblock_one_layer(512)
    ret_cfg[4] = {"fc": {"num_features": 8, "output": 1}}
    return ret_cfg


def get_rp_resnet_18_rp_category_bg():
    ret_cfg = {"as_output": 1}
    ret_cfg[1] = {
            "ResBlock": [
                {"filter_size": 1, "stride": 1, "num_filters": 512, "bn": 1}, 
                {"filter_size": 1, "stride": 1, "num_filters": 512, "bn": 1}, 
                {"filter_size": 1, "stride": 1, "num_filters": 4608, "bn": 1}]
            }
    ret_cfg[2] = ret_cfg[1]
    ret_cfg[3] = ret_cfg[1]
    ret_cfg[4] = {"fc": {"num_features": 8, "output": 1}}
    return ret_cfg


def get_rp_resnet_18(dataset_prefix='rp_imagenet'):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "encode_head", "rp_category"],
            "%s_order" % dataset_prefix: ["encode", "encode_head"],
            }

    # Add encode
    ret_cfg["encode"] = shared_funcs.get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add encode_head
    ret_cfg["encode_head"] = get_rp_resnet_18_encode_head(
            "encode_%i" % num_layers_enc)
    ret_cfg["encode_head_depth"] = 2

    # Add rp_category
    ret_cfg["rp_category"] = get_rp_resnet_18_rp_category()
    ret_cfg["rp_category_depth"] = 4
    return ret_cfg


def get_rp_resnet_18_bg(dataset_prefix='rp_imagenet'):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "encode_head", "rp_category"],
            "%s_order" % dataset_prefix: ["encode", "encode_head"],
            }

    # Add encode
    ret_cfg["encode"] = shared_funcs.get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add encode_head
    ret_cfg["encode_head"] = get_rp_resnet_18_encode_head_bg(
            "encode_%i" % num_layers_enc)
    ret_cfg["encode_head_depth"] = 2

    # Add rp_category
    ret_cfg["rp_category"] = get_rp_resnet_18_rp_category_bg()
    ret_cfg["rp_category_depth"] = 4
    return ret_cfg


def add_depth_head(
        cfg, input_layer, 
        first_upproj_filter=256, no_upproj_layers=4):
    up_proj_cfg = {'input': input_layer}
    upproj_filter = first_upproj_filter
    for layer_idx in range(1, 1 + no_upproj_layers):
        up_proj_cfg[layer_idx] = {
                "UpProj": {
                    "filter_size": 3, 
                    "num_filters": upproj_filter,
                    "bn": 1}}
        upproj_filter = int(upproj_filter / 2)
    cfg['up_projection'] = up_proj_cfg
    cfg['up_projection_depth'] = no_upproj_layers

    depth_out_cfg = {
            'input': 'up_projection_%i' % no_upproj_layers,
            "as_output": 1,
            1: {
                "conv": {
                    "filter_size": 3, 
                    "stride": 2, 
                    "num_filters": 1, 
                    "output": 1, 
                    "upsample": 2}}}
    cfg['depth'] = depth_out_cfg
    cfg['depth_depth'] = 1
    return cfg


def get_ae_resnet_18(dataset_prefix='imagenet', head_dim=128):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "ae_head", "ae_decode", "ae_output"],
            "%s_order" % dataset_prefix: [
                "encode", 
                "ae_head", "ae_decode", "ae_output"],
            }

    # Add encode
    ret_cfg["encode"] = shared_funcs.get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    ret_cfg["ae_head"] = {
            "input": "encode_%i" % num_layers_enc,
            1: {"ae_head": {"dimension": head_dim}},
            }
    ret_cfg["ae_head_depth"] = 1

    ae_decode = {"input": "ae_head_1"}
    for idx in range(1, num_layers_enc):
        curr_encode = ret_cfg["encode"][num_layers_enc + 1 - idx]
        curr_resblock = copy.deepcopy(curr_encode["ResBlock"])
        curr_resblock = list(reversed(curr_resblock))
        for each_conv in curr_resblock:
            each_conv["upsample"] = each_conv["stride"]
        curr_decode = {"ResBlock": curr_resblock}
        ae_decode[idx] = curr_decode
    ret_cfg["ae_decode"] = ae_decode
    ret_cfg["ae_decode_depth"] = num_layers_enc - 1

    ret_cfg["ae_output"] = {
            "input": "ae_decode_%i" % (num_layers_enc - 1),
            "as_output": 1,
            1: {
                "conv": {
                    "filter_size": 3, 
                    "stride": 2, 
                    "num_filters": 3, 
                    "upsample": 2}},
            2: {
                "conv": {
                    "filter_size": 7, 
                    "stride": 2, 
                    "num_filters": 3, 
                    "output": 1,
                    "upsample": 2}},
            }
    ret_cfg["ae_output_depth"] = 2
    return ret_cfg


def get_cpc_resnet_18(dataset_prefix='imagenet'):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode"],
            "%s_order" % dataset_prefix: ["encode"],
            }

    # Add encode
    ret_cfg["encode"] = shared_funcs.get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc
    ret_cfg["encode"]["as_output"] = 1
    return ret_cfg
