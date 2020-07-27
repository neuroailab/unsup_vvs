import unsup_vvs.network_training.saved_setting as saved_setting
import importlib


def get_setting_func(load_setting_func):
    if '.' in load_setting_func:
        all_paths = load_setting_func.split('.')
        module_name = '.'.join(['exp_settings'] + all_paths[:-1])
        load_setting_module = importlib.import_module(module_name)
        load_setting_func = all_paths[-1]
        setting_func = getattr(load_setting_module, load_setting_func)
    else:
        setting_func = getattr(saved_setting, load_setting_func)
    return setting_func


def set_load_setting(args):
    args.loadport = args.nport
    args.load_dbname = args.dbname
    args.load_collname = args.collname
    args.loadexpId = args.expId
    return args
