import json


def infntIN_ir_seed0(args):
    args.load_step = 2215000
    args.load_port = 26001
    args.load_dbname = 'combinet-test'
    args.load_colname = 'combinet'
    args.load_expId = 'inst_infant_18_fx'

    args.identifier = 'infntIN-ir-seed0'
    cfg_kwargs = {
            'network_func': 'get_resnet_18',
            'network_func_kwargs': '{"num_cat": 128}',
            }
    args.cfg_kwargs = json.dumps(cfg_kwargs)
    return args
