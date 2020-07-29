def la_cate(args):
    args.load_step = 160 * 10010
    args.setting_name = 'combine_irla_others.res18_la_cate'
    return args


def la_cate_early(args):
    args.load_step = 150 * 10010
    args.setting_name = 'combine_irla_others.res18_la_cate_early'
    return args


def la_cate_even_early(args):
    args.load_step = 160 * 10010
    args.setting_name = 'combine_irla_others.res18_la_cate_even_early'
    return args


def la_cate_early_enc6(args):
    args.load_step = 110 * 10010
    args.setting_name = 'combine_irla_others.res18_la_cate_early_enc6'
    return args
