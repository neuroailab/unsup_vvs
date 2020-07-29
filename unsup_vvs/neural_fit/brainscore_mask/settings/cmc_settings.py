def cmc_seed0(args):
    args.pt_model = 'la_cmc'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/la_cmc_models/experiments/imagenet_la_cmc/res18_lab_cmc_v1_s0/checkpoints/final.pth.tar'
    args.identifier = 'pt-cmc-seed0'
    return args


def cmc_seed1(args):
    args.pt_model = 'la_cmc'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/la_cmc_models/experiments/imagenet_la_cmc/res18_lab_cmc_v1_s1/checkpoints/final.pth.tar'
    args.identifier = 'pt-cmc-seed1'
    return args


def cmc_seed2(args):
    args.pt_model = 'la_cmc'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/la_cmc_models/experiments/imagenet_la_cmc/res18_lab_cmc_v1_s2/checkpoints/final.pth.tar'
    args.identifier = 'pt-cmc-seed2'
    return args
