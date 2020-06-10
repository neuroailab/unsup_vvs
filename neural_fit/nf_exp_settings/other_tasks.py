from nf_exp_settings.shared_settings import \
        basic_res18_setting


def other_tasks_save_settings(args):
    args.nport = 27009
    args.dbname = 'nf-other-tasks'
    args.colname = 'res18'
    return args


def dp_pbr_shared(args):
    args = basic_res18_setting(args)
    args = other_tasks_save_settings(args)
    args.load_train_setting_func = 'other_tasks.res18_dp_pbr'
    return args


def dp_pbr_basic(args):
    args = dp_pbr_shared(args)
    args.expId = 'dp_pbr'
    args.loadstep = 176 * 10010
    args.train_num_steps = args.loadstep + 30000
    return args


def dp_pbr_basic_60(args):
    args = dp_pbr_shared(args)
    args.expId = 'dp_pbr_60'
    args.loadstep = 60 * 10010
    args.train_num_steps = args.loadstep + 30000
    return args


def dp_pbr_basic_100(args):
    args = dp_pbr_shared(args)
    args.expId = 'dp_pbr_100'
    args.loadstep = 100 * 10010
    args.train_num_steps = args.loadstep + 30000
    return args


def dp_pbr_basic_150(args):
    args = dp_pbr_shared(args)
    args.expId = 'dp_pbr_150'
    args.loadstep = 150 * 10010
    args.train_num_steps = args.loadstep + 30000
    return args


def dp_ps_shared(args):
    args = basic_res18_setting(args)
    args = other_tasks_save_settings(args)
    args.load_train_setting_func = 'other_tasks.res18_dp_ps'
    return args


def dp_ps_basic(args):
    args = dp_ps_shared(args)
    args.expId = 'dp_ps'
    args.loadstep = 140 * 10010
    args.train_num_steps = args.loadstep + 30000
    return args


def rp_basic(args):
    args = basic_res18_setting(args)
    args = other_tasks_save_settings(args)
    args.load_train_setting_func = 'other_tasks.res18_rp_imagenet'
    args.expId = 'rp'
    args.loadstep = 140 * 10010
    args.train_num_steps = args.loadstep + 30000
    return args
