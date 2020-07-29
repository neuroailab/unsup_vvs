import json


def infntIN_cate_seed0(args):
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tpu_ckpts/res18_infant/model.ckpt-687168'
    args.identifier = 'infntIN-cate-seed0'
    cfg_kwargs = {
            'network_func': 'get_resnet_18',
            'network_func_kwargs': '{"num_cat": 246}',
            }
    args.cfg_kwargs = json.dumps(cfg_kwargs)
    return args
