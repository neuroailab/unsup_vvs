from fit_neural_data import get_params_from_arg
import nf_exp_settings.shared_settings as shr_sts
import nf_exp_settings.v1_cadena_setting as v1_sts
import nf_exp_settings.v1v2_fx_setting as v1v2_sts
from nf_cmd_parser import get_parser, load_setting
import os
import copy
import pdb
from tfutils import base
import json


def run_train(args):
    params = get_params_from_arg(args)
    base.train_from_params(**params)


def var6_v4it_wd_swd(args):
    v4it_args = shr_sts.var6_basic_setting_wo_db(copy.deepcopy(args))
    v4it_args.weight_decay = 1e-2
    v4it_args.expId += '_var6_fx_v4it'
    run_train(v4it_args)
    
    v4it_swd_args = shr_sts.var6_basic_setting_wo_db(copy.deepcopy(args))
    v4it_swd_args.weight_decay = 1e-3
    v4it_swd_args.expId += '_var6_fx_v4it_swd'
    run_train(v4it_swd_args)


def v4it_wd_swd(args):
    v4it_args = shr_sts.basic_setting_wo_db(copy.deepcopy(args))
    v4it_args.weight_decay = 1e-2
    v4it_args.expId += '_v4it'
    run_train(v4it_args)
    
    v4it_swd_args = shr_sts.basic_setting_wo_db(copy.deepcopy(args))
    v4it_swd_args.weight_decay = 1e-3
    v4it_swd_args.expId += '_v4it_swd'
    run_train(v4it_swd_args)


def v1_swd(args):
    v1_swd_args = v1_sts.v1_fit_wo_db(copy.deepcopy(args))
    v1_swd_args.weight_decay = 1e-3
    v1_swd_args.expId += '_v1_swd'
    run_train(v1_swd_args)


def v1_wd(args):
    v1_args = v1_sts.v1_fit_wo_db(copy.deepcopy(args))
    v1_args.weight_decay = 1e-2
    v1_args.expId += '_v1'
    run_train(v1_args)


def v1_wd_swd(args):
    v1_wd(args)
    v1_swd(args)


def v1_sswd(args):
    v1_sswd_args = v1_sts.v1_fit_wo_db(copy.deepcopy(args))
    v1_sswd_args.weight_decay = 1e-4
    v1_sswd_args.expId += '_v1_sswd'
    run_train(v1_sswd_args)


def v1_ssswd(args):
    v1_ssswd_args = v1_sts.v1_fit_wo_db(copy.deepcopy(args))
    v1_ssswd_args.weight_decay = 1e-5
    v1_ssswd_args.expId += '_v1_ssswd'
    run_train(v1_ssswd_args)


def v1_swd_sswd(args):
    v1_swd(args)
    v1_sswd(args)


def v1_all_splits_wd_swd_sswd(args):
    for split_id in range(5):
        now_args = copy.deepcopy(args)
        now_args.which_split = 'split_%i' % split_id
        now_args.expId += '_sp%i' % split_id
        v1_wd(now_args)
        v1_swd(now_args)
        v1_sswd(now_args)


def v4it_all_splits_vary_wd(args):
    for split_id in range(5):
        now_args = copy.deepcopy(args)
        now_args.which_split = 'split_%i' % split_id
        now_args.expId += '_sp%i' % split_id
        v4it_wd_swd(now_args)


def var6_v4it_all_splits_vary_wd(args):
    for split_id in range(5):
        now_args = copy.deepcopy(args)
        now_args.which_split = 'split_%i' % split_id
        now_args.expId += '_sp%i' % split_id
        var6_v4it_wd_swd(now_args)


def v4it_chosen_splits_vary_wd(args):
    split_range = range(5)
    if args.pat_func_args is not None:
        split_range = json.loads(args.pat_func_args)['which_splits']
        split_range = split_range.split(',')
        split_range = [int(each_split) for each_split in split_range]
    for split_id in split_range:
        now_args = copy.deepcopy(args)
        now_args.which_split = 'split_%i' % split_id
        now_args.expId += '_sp%i' % split_id
        v4it_wd_swd(now_args)


def v1v4it_wd_swd(args):
    v4it_wd_swd(args)
    v1_wd_swd(args)


def v1v4it_all_splits_vary_wd(args):
    for split_id in range(5):
        now_args = copy.deepcopy(args)
        now_args.which_split = 'split_%i' % split_id
        now_args.expId += '_sp%i' % split_id
        v1_wd(now_args)
        v1_swd(now_args)
        v1_sswd(now_args)
        v4it_wd_swd(now_args)


def v1v4it_bwd(args):
    v4it_bwd_args = shr_sts.basic_setting_wo_db(copy.deepcopy(args))
    v4it_bwd_args.weight_decay = 1e-1
    v4it_bwd_args.expId += '_v4it_bwd'
    run_train(v4it_bwd_args)
    
    v1_bwd_args = v1_sts.v1_fit_wo_db(copy.deepcopy(args))
    v1_bwd_args.weight_decay = 1e-1
    v1_bwd_args.expId += '_v1_bwd'
    run_train(v1_bwd_args)


def v1v4it_bwd_wd_swd(args):
    v1v4it_wd_swd(args)
    v1v4it_bwd(args)


def vary_wd(args):
    wd_args = copy.deepcopy(args)
    wd_args.weight_decay = 1e-2
    run_train(wd_args)

    swd_args = copy.deepcopy(args)
    swd_args.weight_decay = 1e-3
    swd_args.expId += '_swd'
    run_train(swd_args)

    bwd_args = copy.deepcopy(args)
    bwd_args.weight_decay = 1e-1
    bwd_args.expId += '_bwd'
    run_train(bwd_args)


def v1v2_vary_wd(args):
    v1v2_bwd_args = v1v2_sts.v1v2_trandom_50to200_basic_setting_wo_db(
            copy.deepcopy(args))
    v1v2_bwd_args.weight_decay = 1e-1
    v1v2_bwd_args.expId += '_v1v2_bwd'
    run_train(v1v2_bwd_args)

    v1v2_bbwd_args = v1v2_sts.v1v2_trandom_50to200_basic_setting_wo_db(
            copy.deepcopy(args))
    v1v2_bbwd_args.weight_decay = 1.
    v1v2_bbwd_args.expId += '_v1v2_bbwd'
    run_train(v1v2_bbwd_args)

    v1v2_wd_args = v1v2_sts.v1v2_trandom_50to200_basic_setting_wo_db(
            copy.deepcopy(args))
    v1v2_wd_args.weight_decay = 1e-2
    v1v2_wd_args.expId += '_v1v2_wd'
    run_train(v1v2_wd_args)


def v1v2_all_splits_vary_wd(args):
    for split_id in range(5):
        now_args = copy.deepcopy(args)
        now_args.which_split = 'split_%i' % split_id
        now_args.expId += '_sp%i' % split_id
        v1v2_vary_wd(now_args)


def main():
    parser = get_parser()
    args = parser.parse_args()
    args = load_setting(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    pat_func = globals()[args.pat_func]
    pat_func(args)


if __name__ == '__main__':
    main()
