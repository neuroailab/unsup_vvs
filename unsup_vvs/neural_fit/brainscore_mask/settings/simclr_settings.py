def simclr_res18(args):
    args.model_type = 'simclr_model'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18/model.ckpt-311748'
    args.identifier = 'tpu-simclr-res18'
    args.layers = ','.join(['encode_%i' % i for i in range(1, 10)]) # no first conv layer
    return args


def simclr_res18_seed1(args):
    args.model_type = 'simclr_model'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18_2/model.ckpt-311748'
    args.identifier = 'tpu-simclr-res18-s1'
    args.layers = ','.join(['encode_%i' % i for i in range(1, 10)]) # no first conv layer
    return args


def simclr_res18_seed2(args):
    args.model_type = 'simclr_model'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18_3/model.ckpt-311748'
    args.identifier = 'tpu-simclr-res18-s2'
    args.layers = ','.join(['encode_%i' % i for i in range(1, 10)]) # no first conv layer
    return args
