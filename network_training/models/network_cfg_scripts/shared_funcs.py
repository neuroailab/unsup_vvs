def get_category(num_cat, input_name):
    return {
            "as_output":1,
            "input":input_name,
            1:{
                "pool":{
                    "filter_size":7, 
                    "type":"avg", 
                    "stride":1, 
                    "padding":"VALID"}},
            2: {"fc": {"num_features": num_cat, "output": 1}}
            }


def get_category_pool_stride7(num_cat, input_name):
    return {
            "as_output":1,
            "input":input_name,
            1:{
                "pool":{
                    "filter_size": 7, 
                    "type": "avg", 
                    "stride": 7, 
                    "padding": "VALID"}},
            2: {"fc": {"num_features": num_cat, "output": 1}}
            }


def get_resnet_layer1():
    return {
            "conv":{
                "filter_size":7, 
                "stride":2, 
                "num_filters":64, 
                "bn":1, 
                "padding":"VALID"}, 
            "pool":{
                "filter_size":3, 
                "type":"max", 
                "stride":2}
            }


def get_resnet_basicblock_one_layer(num_unit):
    return {
            "ResBlock":[
                {
                    "filter_size":3, 
                    "stride":1, 
                    "num_filters":num_unit,
                    "bn":1}, 
                {
                    "filter_size":3, 
                    "stride":1, 
                    "num_filters":num_unit, 
                    "bn":1}]
            }


def get_resnet_bottleblock_one_layer(num_unit):
    return {
            "ResBlock":[
                {
                    "filter_size":1, 
                    "stride":1, 
                    "num_filters":num_unit,
                    "bn":1}, 
                {
                    "filter_size":3, 
                    "stride":1, 
                    "num_filters":num_unit,
                    "bn":1}, 
                {
                    "filter_size":1, 
                    "stride":1, 
                    "num_filters":num_unit*4,
                    "bn":1}]
            }


def get_resnet_basicblock(
        num_layers=[3, 4, 6, 3], 
        num_units=[64, 128, 256, 512],
        ):
    # Build ResNet network configs according to the setting
    # Default parameter is for resnet34
    # Returns a list of dictionaries
    ret_list = []
    first_flag = True
    for num_layer, num_unit in zip(num_layers, num_units):
        now_first = True
        for _ in range(num_layer):
            curr_layer = get_resnet_basicblock_one_layer(num_unit)
            if now_first and not first_flag:
                curr_layer['ResBlock'][0]['stride'] = 2
            ret_list.append(curr_layer)
            now_first = False
            first_flag = False
    return ret_list


def get_resnet_bottleblock(
        num_layers=[3, 4, 6, 3], 
        num_units=[64, 128, 256, 512],
        stride2_style='default',
        ):
    # Build ResNet network configs according to the setting
    # Default parameter is for resnet50
    # Returns a list of dictionaries
    ret_list = []
    for idx_block, (num_layer, num_unit) \
            in enumerate(zip(num_layers, num_units)):
        for idx_layer in range(num_layer):
            curr_layer = get_resnet_bottleblock_one_layer(num_unit)

            if stride2_style == 'default':
                if idx_layer == 0 and idx_block > 0:
                    curr_layer['ResBlock'][1]['stride'] = 2
            elif stride2_style == '101_3blk_siming':
                if idx_layer == num_layer-1:
                    curr_layer['ResBlock'][0]['stride'] = 2
            else:
                raise NotImplementedError

            ret_list.append(curr_layer)
    return ret_list


def get_encode_resnet(
        func=get_resnet_basicblock,
        func_kwargs={},
        ):
    ret_cfg = {}
    # Get layer1
    ret_cfg[1] = get_resnet_layer1()
    other_layers = func(**func_kwargs)
    for idx, curr_layer in enumerate(other_layers):
        ret_cfg[2 + idx] = curr_layer
    return ret_cfg


def get_resnet_18(num_cat=1000):
    ret_cfg = {
            "BATCH_SIZE": 8,
            "QUEUE_CAP": 2560,
            "network_list": ["encode", "category"],
            "imagenet_order": ["encode", "category"],
            }

    # Add encode here
    ret_cfg["encode"] = get_encode_resnet(
            func_kwargs={'num_layers': [2, 2, 2, 2]})
    num_layers_enc = len(ret_cfg["encode"].keys())
    ret_cfg["encode_depth"] = num_layers_enc

    # Add category here
    ret_cfg["category"] = get_category(num_cat, "encode_%i" % num_layers_enc)
    ret_cfg["category_depth"] = 2

    return ret_cfg
