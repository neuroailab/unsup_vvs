from unsup_vvs.network_training.models.network_cfg_scripts.vgg_funcs import get_vgg16
from unsup_vvs.network_training.models.network_cfg_scripts.shared_funcs import get_category, \
        get_category_pool_stride7, get_resnet_layer1, \
        get_resnet_basicblock_one_layer, get_resnet_bottleblock_one_layer, \
        get_resnet_basicblock, get_resnet_bottleblock, \
        get_encode_resnet, get_resnet_18


def get_resnet_18_wun(num_cat=1000):
    ret_cfg = get_resnet_18(num_cat)
    ret_cfg["imagenet_un_order"] = ret_cfg["imagenet_order"]
    return ret_cfg


def get_resnet_18_saycam(num_cat=1000):
    ret_cfg = get_resnet_18(num_cat)
    ret_cfg["saycam_order"] = ret_cfg["imagenet_order"]
    return ret_cfg


def get_res18_clstr_un_mt(num_cat=128):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "cons", "memory"],
            "imagenet_order": ["encode", "cons", "memory"],
            "imagenet_un_order": ["encode", "cons", "memory"],
            }

    # Add encode here
    ret_cfg["encode"] = get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add memory here
    ret_cfg["memory"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["memory_depth"] = 2

    # Add cons here
    ret_cfg["cons"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["cons_depth"] = 2
    return ret_cfg


def get_resnet_18_inst_and_cate(num_cat=1000, dim_memory=128):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "category", "memory"],
            "imagenet_order": ["encode", "category", "memory"],
            }

    # Add encode here
    ret_cfg["encode"] = get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add category here
    ret_cfg["category"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["category_depth"] = 2

    # Add memory here
    # Memory should be the last module
    ret_cfg["memory"] = get_category(dim_memory, "encode_%i" % num_layers_enc)
    ret_cfg["memory_depth"] = 2

    return ret_cfg


def get_resnet_18_inst_and_cate_sep(num_cat=1000, dim_memory=128):
    ret_cfg = get_resnet_18_inst_and_cate(num_cat, dim_memory)
    ret_cfg["imagenet_un_order"] = ["encode", "memory"]
    ret_cfg["imagenet_order"] = ["encode", "category"]
    return ret_cfg


def get_resnet_18_inst_and_cate_early_memory(num_cat=1000, dim_memory=128):
    ret_cfg = get_resnet_18_inst_and_cate(num_cat, dim_memory)
    ret_cfg["memory"] = get_category(dim_memory, "encode_8")
    return ret_cfg


def get_resnet_18_inst_and_cate_even_early_memory(num_cat=1000, dim_memory=128):
    ret_cfg = get_resnet_18_inst_and_cate(num_cat, dim_memory)
    ret_cfg["memory"] = get_category_pool_stride7(dim_memory, "encode_7")
    return ret_cfg


def get_resnet_18_inst_and_cate_memory_enc6(num_cat=1000, dim_memory=128):
    ret_cfg = get_resnet_18_inst_and_cate(num_cat, dim_memory)
    ret_cfg["memory"] = get_category_pool_stride7(dim_memory, "encode_6")
    return ret_cfg


def get_resnet_18_cate_inst_branch2(num_cat=1000, dim_memory=128):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "category", "memory"],
            "imagenet_order": ["encode", "category"],
            "imagenet_branch2_order": ["encode", "memory"],
            }

    # Add encode here
    ret_cfg["encode"] = get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add category here
    ret_cfg["category"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["category_depth"] = 2

    # Add memory here
    # Memory should be the last module
    ret_cfg["memory"] = get_category(dim_memory, "encode_%i" % num_layers_enc)
    ret_cfg["memory_depth"] = 2

    return ret_cfg


def get_resnet_34(num_cat=1000):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "category"],
            "imagenet_order": ["encode", "category"],
            }

    # Add encode here
    ret_cfg["encode"] = get_encode_resnet()
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add category here
    ret_cfg["category"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["category_depth"] = 2

    return ret_cfg


def get_resnet_50(num_cat=1000):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "category"],
            "imagenet_order": ["encode", "category"],
            }

    # Add encode here
    ret_cfg["encode"] = get_encode_resnet(func=get_resnet_bottleblock)
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add category here
    ret_cfg["category"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["category_depth"] = 2

    return ret_cfg


def get_mean_teacher_resnet_50(num_cat=1000):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "category", "cons"],
            "imagenet_order": ["encode", "category", "cons"],
            "imagenet_un_order": ["encode", "category", "cons"],
            }

    # Add encode here
    ret_cfg["encode"] = get_encode_resnet(func=get_resnet_bottleblock)
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add category here
    ret_cfg["category"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["category_depth"] = 2

    # Add cons here
    ret_cfg["cons"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["cons_depth"] = 2

    return ret_cfg


def get_resnet_101_3blk(num_cat=1000):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "category"],
            "imagenet_order": ["encode", "category"],
            }

    # Add encode here
    ret_cfg["encode"] = get_encode_resnet(
            func=get_resnet_bottleblock,
            func_kwargs={
                'num_layers': [3, 4, 23, 0],
                'stride2_style': '101_3blk_siming',
                },
            )
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add category here
    ret_cfg["category"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["category_depth"] = 2

    return ret_cfg


def get_mean_teacher_resnet_18(num_cat=1000):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "category", "cons"],
            "imagenet_order": ["encode", "category", "cons"],
            "imagenet_un_order": ["encode", "category", "cons"],
            }

    # Add encode here
    ret_cfg["encode"] = get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add category here
    ret_cfg["category"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["category_depth"] = 2

    # Add cons here
    ret_cfg["cons"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["cons_depth"] = 2

    return ret_cfg


def get_mean_teacher_and_inst_resnet_18(num_cat=1000, dim_memory=128):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "category", "cons", "memory"],
            "imagenet_order": ["encode", "category", "cons", "memory"],
            "imagenet_un_order": ["encode", "category", "cons", "memory"],
            }

    # Add encode here
    ret_cfg["encode"] = get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add category here
    ret_cfg["category"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["category_depth"] = 2

    # Add cons here
    ret_cfg["cons"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["cons_depth"] = 2

    # Add memory here
    # Memory should be the last module
    ret_cfg["memory"] = get_category(dim_memory, "encode_%i" % num_layers_enc)
    ret_cfg["memory_depth"] = 2

    return ret_cfg


def get_fc18():
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "category"],
            "imagenet_order": ["encode", "category"],
            }

    encode = {}
    for idx in range(1, 18):
        encode[idx] = {"fc": {"num_features": 1000}}
    ret_cfg['encode_depth'] = 17
    ret_cfg['encode'] = encode

    ret_cfg['category'] = {
            "as_output":1,
            "input": "encode_17",
            1: {"fc": {"num_features": 1000, "output": 1}},
            }
    ret_cfg['category_depth'] = 1
    return ret_cfg
