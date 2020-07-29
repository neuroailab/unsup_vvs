def get_encode_vgg16():
    num_filter_list = [64]*2 + [128]*2 + [256]*3 + [512]*6
    pool_layer_list = [2, 4, 7, 10]

    ret_cfg = {}
    for now_layer, num_filter in zip(range(1, 14), num_filter_list):
        curr_layer = {}
        curr_layer["conv"] = {
                "filter_size": 3, 
                "stride": 1, 
                "bn": 1,
                "num_filters": num_filter,
                }
        if now_layer in pool_layer_list:
            curr_layer["pool"] = {
                    "filter_size": 2, 
                    "type": "max", 
                    "stride": 2
                    }
        ret_cfg[now_layer] = curr_layer
    return ret_cfg


def get_category_vggnet(num_cat, input_name):
    return {
            "as_output": 1,
            "input": input_name,
            1: {
                "pool":{
                    "filter_size": 2, 
                    "type": "max", 
                    "stride": 2, 
                    "padding": "VALID"
                    }
                },
            2: {"fc": {"num_features": 4096, "dropout": 0.5}},
            3: {"fc": {"num_features": 4096, "dropout": 0.5}},
            4: {"fc": {"num_features": num_cat, "output": 1}},
            }


def get_vgg16(num_cat=1000):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "category"],
            "imagenet_order": ["encode", "category"],
            }

    # Add encode here
    ret_cfg["encode"] = get_encode_vgg16()
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add category here
    ret_cfg["category"] = get_category_vggnet(
            num_cat, 
            "encode_%i" % num_layers_enc)
    ret_cfg["category_depth"] = 4
    return ret_cfg
