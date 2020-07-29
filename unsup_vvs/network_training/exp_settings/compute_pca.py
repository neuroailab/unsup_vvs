import exp_settings.utils as set_utils
import numpy as np
import pdb


def pca_save(args):
    args.nport = 27007
    args.dbname = 'pca'
    args.collname = 'pca'
    return args


def pca_from_setting_func(args):
    setting_func = set_utils.get_setting_func(args.pca_load_setting_func)
    args = setting_func(args)
    args = set_utils.set_load_setting(args)
    args.expId = '{db_name}_{col_name}_{exp_name}'.format(
            db_name=args.dbname,
            col_name=args.collname,
            exp_name=args.expId)
    args = pca_save(args)
    args.do_pca = True
    args.valinum = int(np.ceil(1000.0 / args.valbatchsize))
    return args
