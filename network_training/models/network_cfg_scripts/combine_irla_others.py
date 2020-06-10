import models.network_cfg_scripts.shared_funcs as shared_funcs
import models.network_cfg_scripts.other_tasks as other_tasks


def get_rp_inst_resnet_18(dataset_prefix='rp_imagenet', rp_layer_offset=0):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "encode_head", "rp_category", "category"],
            "%s_order" % dataset_prefix: ["encode", "encode_head"],
            "imagenet_order": ["encode", "category"],
            }

    # Add encode
    ret_cfg["encode"] = shared_funcs.get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add encode_head
    ret_cfg["encode_head"] = other_tasks.get_rp_resnet_18_encode_head(
            "encode_%i" % (num_layers_enc - rp_layer_offset))
    ret_cfg["encode_head_depth"] = 2

    # Add rp_category
    ret_cfg["rp_category"] = other_tasks.get_rp_resnet_18_rp_category()
    ret_cfg["rp_category_depth"] = 4

    # Add category here
    ret_cfg["category"] = shared_funcs.get_category(
            128, "encode_%i" % num_layers_enc)
    ret_cfg["category_depth"] = 2
    return ret_cfg


def get_resnet_18_fx_encd1():
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["inst_encode", "inst_category"],
            "imagenet_order": ["inst_encode", "inst_category"],
            }

    # Add inst_encode here
    ret_cfg["inst_encode"] = shared_funcs.get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["inst_encode"].keys())
    ret_cfg["inst_encode_depth"] = num_layers_enc

    # Add inst_category here
    ret_cfg["inst_category"] = shared_funcs.get_category(
            128, "inst_encode_%i" % num_layers_enc)
    ret_cfg["inst_category_depth"] = 2

    ret_cfg["inst_encode"][1]["conv"]["trainable"] = 0
    return ret_cfg


def get_resnet_18_fx_encdX(no_layers_fixed=3):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["inst_encode", "inst_category"],
            "imagenet_order": ["inst_encode", "inst_category"],
            }

    # Add inst_encode here
    ret_cfg["inst_encode"] = shared_funcs.get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["inst_encode"].keys())
    ret_cfg["inst_encode_depth"] = num_layers_enc

    # Add inst_category here
    ret_cfg["inst_category"] = shared_funcs.get_category(
            128, "inst_encode_%i" % num_layers_enc)
    ret_cfg["inst_category_depth"] = 2

    assert no_layers_fixed >= 1
    ret_cfg["inst_encode"][1]["conv"]["trainable"] = 0
    ret_cfg["inst_encode"][1]["conv"]["bn_trainable"] = 0
    for idx_layer in range(2, no_layers_fixed + 1):
        ret_cfg["inst_encode"][idx_layer]["ResBlock_trainable"] = 0
        ret_cfg["inst_encode"][idx_layer]["ResBlock_bn_trainable"] = 0
    return ret_cfg


def get_resnet18_inst_and_depth(
        which_encode_to_depth=9, 
        depth_dataset_list=['pbrnet', 'scenenet']):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "up_projection", "depth", "category"],
            "imagenet_order": ["encode", "category"]}
    for depth_dataset in depth_dataset_list:
        order_name = '%s_order' % depth_dataset
        ret_cfg[order_name] = ["encode", "up_projection", "depth"]

    # Add encode
    ret_cfg["encode"] = shared_funcs.get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add category here
    ret_cfg["category"] = shared_funcs.get_category(
            128, "encode_%i" % num_layers_enc)
    ret_cfg["category_depth"] = 2

    if which_encode_to_depth == 9:
        ret_cfg = other_tasks.add_depth_head(
                ret_cfg, 'encode_%i' % which_encode_to_depth)
    elif which_encode_to_depth == 7:
        ret_cfg = other_tasks.add_depth_head(
                ret_cfg, 'encode_%i' % which_encode_to_depth,
                first_upproj_filter=128, no_upproj_layers=3)
    else:
        raise NotImplementedError()
    return ret_cfg


def get_resnet_vgglike_layer12(strides=[1,2]):
    ret_cfg = {}
    ret_cfg[1] = {
            "conv":{
                "filter_size":3, 
                "stride":strides[0], 
                "num_filters":64, 
                "bn":1, 
                "padding":"VALID"}
            }
    ret_cfg[2] = {
            "conv":{
                "filter_size":3, 
                "stride":strides[1], 
                "num_filters":64, 
                "bn":1, 
                "padding":"VALID"}, 
            "pool":{
                "filter_size":3, 
                "type":"max", 
                "stride":2}
            }
    return ret_cfg


def get_resnet_vgglike_19(num_cat=1000, **kwargs):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "category"],
            "imagenet_order": ["encode", "category"],
            }

    ret_cfg["encode"] = get_resnet_vgglike_layer12(**kwargs)
    other_layers = shared_funcs.get_resnet_basicblock(num_layers=[2, 2, 2, 2])
    for idx, curr_layer in enumerate(other_layers):
        ret_cfg["encode"][3 + idx] = curr_layer
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add category here
    ret_cfg["category"] = shared_funcs.get_category(
            num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["category_depth"] = 2
    return ret_cfg
