def la_cmc_seed0(args):
    args.pt_model = 'la_cmc'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/la_cmc_models/experiments/imagenet_la_cmc/res18_lab_cmc+la_v1_s0/checkpoints/final.pth.tar'
    args.identifier = 'pt-la-cmc-seed0'
    return args


def la_cmc_seed1(args):
    args.pt_model = 'la_cmc'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/la_cmc_models/experiments/imagenet_la_cmc/res18_lab_cmc+la_v1_s1/checkpoints/final.pth.tar'
    args.identifier = 'pt-la-cmc-seed1'
    return args


def la_cmc_seed2(args):
    args.pt_model = 'la_cmc'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/la_cmc_models/experiments/imagenet_la_cmc/res18_lab_cmc+la_v1_s2/checkpoints/final.pth.tar'
    args.identifier = 'pt-la-cmc-seed2'
    return args
