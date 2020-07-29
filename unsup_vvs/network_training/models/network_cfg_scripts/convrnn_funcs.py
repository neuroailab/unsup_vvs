import shared_funcs


def get_convrnn_encoder():
    return {'convrnn': {}}


def convrnn_cate(num_cat=1000):
    ret_cfg = {
            "network_list": ["encode", "category"],
            "imagenet_order": ["encode", "category"],
            }

    # Add encode here
    ret_cfg["encode"] = get_convrnn_encoder()

    # Add category here
    ret_cfg["category"] = shared_funcs.get_category(
            num_cat, 
            "conv7")
    ret_cfg["category_depth"] = 2
    return ret_cfg
