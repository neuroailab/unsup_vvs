import json
import os
import copy
import importlib
import pdb
import unsup_vvs.network_training.models.network_cfg_funcs as network_cfg_funcs


def postprocess_config(cfg):
    cfg = copy.deepcopy(cfg)
    for k in cfg.get("network_list", ["decode", "encode"]):
        if k in cfg:
            ks = copy.deepcopy(list(cfg[k].keys()))
            for _k in ks:
                if _k.isdigit():
                    cfg[k][int(_k)] = cfg[k].pop(_k)
    return cfg


def get_network_cfg(args, configdir='network_configs'):
    if not args.network_func:
        pathconfig = args.pathconfig
        if not os.path.exists(pathconfig):
            pathconfig = os.path.join(configdir, pathconfig)
        cfg_initial = postprocess_config(json.load(open(pathconfig)))
    else:
        if callable(args.network_func):
            network_func = args.network_func
        elif '.' not in args.network_func:
            network_func = getattr(network_cfg_funcs, args.network_func)
        else:
            all_paths = args.network_func.split('.')
            module_name = '.'.join(['models', 'network_cfg_scripts'] + all_paths[:-1])
            network_module = importlib.import_module(module_name)
            network_func_name = all_paths[-1]
            network_func = getattr(network_module, network_func_name)

        network_func_kwargs = {}
        if args.network_func_kwargs:
            network_func_kwargs = json.loads(args.network_func_kwargs)

        cfg_initial = network_func(**network_func_kwargs)
    return cfg_initial


class ConfigParser(object):
    """
    Parsing config for network structure
    """
    def __init__(self, cfg_initial):
        self.cfg_initial = cfg_initial

    def get_dataset_order(self, dataset_prefix):
        curr_order = '%s_order' % dataset_prefix
        assert curr_order in self.cfg_initial, "Module order not found!"
        network_order = self.cfg_initial.get(curr_order)
        return network_order

    def set_current_module(self, module_name):
        self.module_name = module_name

    def get_input_from_other_modules(self):
        return self.cfg_initial[self.module_name].get("input", None)

    def get_var_name(self):
        tmp_dict = self.cfg_initial[self.module_name]
        return tmp_dict.get('var_name', self.module_name)

    def get_var_offset(self):
        tmp_dict = self.cfg_initial[self.module_name]
        return tmp_dict.get('var_offset', 0)
    
    def get_bn_var_name(self):
        tmp_dict = self.cfg_initial[self.module_name]
        return tmp_dict.get('bn_var_name', '')

    def get_module_depth(self):
        val = None
        key_want = self.module_name
        cfg = self.cfg_initial
    
        depth_key = '%s_depth' % key_want
        if depth_key in cfg:
            val = cfg[depth_key]
        elif key_want in cfg:
            val = max([v for v in cfg[key_want].keys() if isinstance(v, int)])

        assert val, "Depth for module %s not specified" % self.module_name
        return val

    def set_curr_layer(self, curr_layer):
        self.curr_layer = curr_layer

    def get_bypass_add(self):
        tmp_dict = self.cfg_initial[self.module_name][self.curr_layer]
        ret_val = tmp_dict.get('bypass_add', None)
        return ret_val

    def get_bypass_light(self):
        val = None
        cfg = self.cfg_initial
        key_want = self.module_name

        if key_want in cfg and (self.curr_layer in cfg[key_want]):
            if 'bypass' in cfg[key_want][self.curr_layer]:
                val = cfg[key_want][self.curr_layer]['bypass']
        return val

    def get_whether_resBlock(self):
        tmp_dict = self.cfg_initial[self.module_name][self.curr_layer]
        return 'ResBlock' in tmp_dict

    def get_resBlock_conv_settings(self):
        tmp_dict = self.cfg_initial[self.module_name][self.curr_layer]
        return tmp_dict['ResBlock']

    def get_resBlock_trainable_settings(self):
        trainable_kwargs = {}
        tmp_dict = self.cfg_initial[self.module_name][self.curr_layer]
        if 'ResBlock_trainable' in tmp_dict:
            trainable_kwargs['sm_trainable'] \
                    = tmp_dict['ResBlock_trainable']==1
        if 'ResBlock_bn_trainable' in tmp_dict:
            trainable_kwargs['sm_bn_trainable'] \
                    = tmp_dict['ResBlock_bn_trainable']==1
        return trainable_kwargs

    def whether_do_conv(self):
        ret_val = False
        if 'conv' in self.cfg_initial[self.module_name][self.curr_layer]:
            ret_val = True
        return ret_val

    def get_conv_kwargs(self):
        cfg = self.cfg_initial
        conv_config = cfg[self.module_name][self.curr_layer]['conv']
        conv_kwargs = {
                'out_shape': conv_config['num_filters'],
                'ksize': conv_config['filter_size'],
                'stride': conv_config['stride'],
                'whetherBn': conv_config.get('bn', 0) == 1,
                'dilat': conv_config.get('dilat', 1),
                }

        trainable = conv_config.get("trainable", 1) == 1
        conv_kwargs["trainable"] = trainable

        bn_trainable = conv_config.get("bn_trainable", 1) == 1
        conv_kwargs["sm_bn_trainable"] = bn_trainable

        activation = 'relu'
        if conv_config.get("output", 0):
            activation = None
        conv_kwargs['activation'] = activation
        conv_kwargs['bias'] = 0
        return conv_kwargs, conv_config

    def get_whether_unpool(self):
        val = False
        if 'unpool' in self.cfg_initial[self.module_name][self.curr_layer]:
            val = True
        return False

    def get_unpool_scale(self):
        cfg = self.cfg_initial
        unpool_scale = cfg[self.module_name][self.curr_layer]['unpool']['scale']
        return unpool_scale

    def get_whether_fc(self):
        ret_val = False
        cfg = self.cfg_initial
        if 'fc' in cfg[self.module_name][self.curr_layer]:
            ret_val = True
        return ret_val

    def get_fc_config(self):
        cfg = self.cfg_initial
        fc_config = cfg[self.module_name][self.curr_layer]['fc']
        return fc_config

    def get_whether_pool(self):
        ret_val = False
        cfg = self.cfg_initial
        if 'pool' in cfg[self.module_name][self.curr_layer]:
            ret_val = True
        return ret_val

    def get_pool_config(self):
        cfg = self.cfg_initial
        pool_config = cfg[self.module_name][self.curr_layer]['pool']
        return pool_config

    def get_whether_upproj(self):
        cfg = self.cfg_initial
        tmp_dict = cfg[self.module_name][self.curr_layer]
        return 'UpProj' in tmp_dict

    def get_upproj_settings(self):
        cfg = self.cfg_initial
        tmp_dict = cfg[self.module_name][self.curr_layer]
        return tmp_dict['UpProj']

    def get_whether_bn(self):
        ret_val = False
        cfg = self.cfg_initial
        if 'bn' in cfg[self.module_name][self.curr_layer]:
            ret_val = True
        return ret_val

    def get_as_output(self):
        ret_val = self.cfg_initial[self.module_name].get('as_output', 0)
        return ret_val

    def get_convrnn_params(self):
        ret_val = self.cfg_initial[self.module_name].get('convrnn', None)
        return ret_val

    def get_whether_ae_head(self):
        ret_val = 'ae_head' in self.cfg_initial[self.module_name]\
                                               [self.curr_layer]
        return ret_val

    def get_ae_head_dim(self):
        ret_val = self.cfg_initial[self.module_name][self.curr_layer]\
                                  ['ae_head']['dimension']
        return ret_val
