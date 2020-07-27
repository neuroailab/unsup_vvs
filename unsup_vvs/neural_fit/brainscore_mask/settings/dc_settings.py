def dc_seed0(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np/checkpoint.pth.tar'
    args.identifier = 'pt-dc-seed0'
    return args


def dc_seed1(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np_seed0/checkpoint.pth.tar'
    args.identifier = 'pt-dc-seed1'
    return args


def dc_seed2(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np_seed1/checkpoint.pth.tar'
    args.identifier = 'pt-dc-seed2'
    return args


def dc_seed1_ep2(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np_seed0/checkpoints/checkpoint_2.pth.tar'
    args.identifier = 'pt-dc-seed1-ep2'
    return args


def dc_seed1_ep4(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np_seed0/checkpoints/checkpoint_4.pth.tar'
    args.identifier = 'pt-dc-seed1-ep4'
    return args


def dc_seed1_ep6(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np_seed0/checkpoints/checkpoint_6.pth.tar'
    args.identifier = 'pt-dc-seed1-ep6'
    return args


def dc_seed1_ep8(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np_seed0/checkpoints/checkpoint_8.pth.tar'
    args.identifier = 'pt-dc-seed1-ep8'
    return args


def dc_seed1_ep10(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np_seed0/checkpoints/checkpoint_10.pth.tar'
    args.identifier = 'pt-dc-seed1-ep10'
    return args


def dc_seed1_ep20(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np_seed0/checkpoints/checkpoint_20.pth.tar'
    args.identifier = 'pt-dc-seed1-ep20'
    return args


def dc_seed1_ep40(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np_seed0/checkpoints/checkpoint_40.pth.tar'
    args.identifier = 'pt-dc-seed1-ep40'
    return args


def dc_seed1_ep70(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np_seed0/checkpoints/checkpoint_70.pth.tar'
    args.identifier = 'pt-dc-seed1-ep70'
    return args
