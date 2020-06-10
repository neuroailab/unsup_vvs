def vie_infant_3dresnet(args):
    args.prep_type = 'no_prep'
    step_num = 1800000
    args.load_from_ckpt = '/mnt/fs4/chengxuz/tfutils_ckpts/vd_unsup_fx/infant/vd_3dresnet_k10k/checkpoint-%i' % step_num
    args.identifier = 'vie-infant-3dresnet'
    args.model_type = 'vd_inst_model:3dresnet'
    args.layers = ','.join(['encode_%i' % i for i in range(2, 11)]) # no first conv layer
    return args


def vie_infant_3dresnet_full(args):
    args.prep_type = 'no_prep'
    step_num = 1200000
    args.load_from_ckpt = '/mnt/fs4/chengxuz/brainscore_model_caches/vd_unsup_fx/infant/vd_3dresnet_full_v2/checkpoint-%i' % step_num
    args.identifier = 'vie-infant-3dresnet-full'
    args.model_type = 'vd_inst_model:3dresnet_full'
    args.layers = ','.join(['encode_%i' % i for i in range(2, 11)]) # no first conv layer
    return args


def vie_infant_3dsingle_resnet_full(args):
    args.prep_type = 'no_prep'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/brainscore_model_caches/vd_unsup_fx/infant/vd_3d_single/model'
    args.identifier = 'vie-infant-3dsingle-resnet-full'
    args.model_type = 'vd_inst_model:vanilla3d_single'
    args.layers = ','.join(['encode_%i' % i for i in range(2, 11)]) # no first conv layer
    return args


def vie_infant_single(args):
    args.prep_type = 'no_prep'
    step_num = 1400000
    args.load_from_ckpt = '/mnt/fs4/chengxuz/brainscore_model_caches/vd_unsup_fx/infant/vd_ctl_infant/checkpoint-%i' % step_num
    args.identifier = 'vie-infant-single'
    args.model_type = 'inst_model'
    args.layers = ','.join(['encode_%i' % i for i in range(2, 11)]) # no first conv layer
    return args
