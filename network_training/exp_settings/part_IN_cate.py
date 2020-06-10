from exp_settings.shared_settings import basic_res18, bs_256_less_save_setting


def save_settings(args):
    args.nport = 26001
    args.dbname = 'part_IN_cate'
    args.collname = 'res18'
    return args


def part2_cate_res18(args):
    args = basic_res18(args)
    args = save_settings(args)
    args.expId = 'p02'
    args.whichimagenet = 'part2_balanced'
    args.train_num_steps = 401000
    return args


def part2_cate_res18_s1(args):
    args = part2_cate_res18(args)
    args.expId = 'p02_s1'
    args.seed = 1
    return args


def part2_cate_res18_s2(args):
    args = part2_cate_res18(args)
    args.expId = 'p02_s2'
    args.seed = 2
    return args


def part3_cate_res18_s1(args):
    args = basic_res18(args)
    args = save_settings(args)
    args.seed = 1
    args.expId = 'p03_s1'
    args.whichimagenet = 'part3'
    args.train_num_steps = 401000
    return args


def part3_cate_res18_s2(args):
    args = part3_cate_res18_s1(args)
    args.expId = 'p03_s2'
    args.seed = 2
    return args


def part4_cate_res18(args):
    args = part2_cate_res18(args)
    args.expId = 'p04'
    args.whichimagenet = 'part4_balanced'
    return args


def part4_cate_res18_s1(args):
    args = part4_cate_res18(args)
    args.expId = 'p04_s1'
    args.seed = 1
    return args


def part4_cate_res18_s2(args):
    args = part4_cate_res18(args)
    args.expId = 'p04_s2'
    args.seed = 2
    return args


def part5_cate_res18_s1(args):
    args = basic_res18(args)
    args = save_settings(args)
    args.seed = 1
    args.expId = 'p05_s1'
    args.whichimagenet = 'part5'
    args.train_num_steps = 401000
    return args


def part5_cate_res18_s2(args):
    args = part5_cate_res18_s1(args)
    args.expId = 'p05_s2'
    args.seed = 2
    return args


def part6_cate_res18(args):
    args = part2_cate_res18(args)
    args.expId = 'p06'
    args.whichimagenet = 'part6_balanced'
    return args


def part6_cate_res18_s1(args):
    args = part6_cate_res18(args)
    args.expId = 'p06_s1'
    args.seed = 1
    return args


def part6_cate_res18_s2(args):
    args = part6_cate_res18(args)
    args.expId = 'p06_s2'
    args.seed = 2
    return args


def part10_cate_res18_s1(args):
    args = basic_res18(args)
    args = save_settings(args)
    args.seed = 1
    args.expId = 'p10_s1'
    args.whichimagenet = 'part10'
    args.train_num_steps = 401000
    return args


def part10_cate_res18_s2(args):
    args = part10_cate_res18_s1(args)
    args.expId = 'p10_s2'
    args.seed = 2
    return args


def part20_cate_res18(args):
    args = part2_cate_res18(args)
    args.expId = 'p20'
    args.whichimagenet = 'part20_balanced'
    return args


def part20_cate_res18_s1(args):
    args = part20_cate_res18(args)
    args.expId = 'p20_s1'
    args.seed = 1
    return args


def part20_cate_res18_s2(args):
    args = part20_cate_res18(args)
    args.expId = 'p20_s2'
    args.seed = 2
    return args


def part50_cate_res18(args):
    args = part2_cate_res18(args)
    args.expId = 'p50'
    args.whichimagenet = 'part50_balanced'
    return args


def part50_cate_res18_s1(args):
    args = part50_cate_res18(args)
    args.expId = 'p50_s1'
    args.seed = 1
    return args


def part50_cate_res18_s2(args):
    args = part50_cate_res18(args)
    args.expId = 'p50_s2'
    args.seed = 2
    return args


def part1_cate_res18(args):
    args = part2_cate_res18(args)
    args.expId = 'p01'
    args.whichimagenet = 'part1_balanced'
    return args


def part1_cate_res18_s1(args):
    args = part1_cate_res18(args)
    args.expId = 'p01_s1'
    args.seed = 1
    return args


def part1_cate_res18_s2(args):
    args = part1_cate_res18(args)
    args.expId = 'p01_s2'
    args.seed = 2
    return args
