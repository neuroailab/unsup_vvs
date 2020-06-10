import os
import sys
import pdb
import numpy as np
from tfutils import base
try:
    import pickle
except:
    import cPickle as pickle
import brainscore.benchmarks as bench
import cleaned_network_builder as net_builder
from nf_cmd_parser import load_setting
from nf_bs_cmd_parser import get_brainscore_parser
from data_for_brainscore import dataset_func
from nf_brain_score import build_wrapper, build_pls_wrapper


def get_save_params_from_arg(args):
    cache_dir = os.path.join(
            args.cacheDirPrefix, '.tfutils',
            'neuralfit-test', 'neuralfit', args.expId)
    save_to_gfs = []
    save_params = {
            'host': 'localhost',
            'port': args.nport,
            'dbname': args.dbname,
            'collname': args.colname,
            'exp_id': args.expId,

            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': args.fre_metric,
            'save_valid_freq': args.fre_valid,
            'save_filters_freq': args.fre_filter,
            'cache_filters_freq': args.fre_filter,
            'cache_dir': cache_dir,
            'save_to_gfs': save_to_gfs}
    return save_params


def get_load_params_from_arg(args):
    loadport = args.nport
    if not args.loadport is None:
        loadport = args.loadport
    loadexpId = args.expId
    if not args.loadexpId is None:
        loadexpId = args.loadexpId
    load_query = None
    if not args.loadstep is None:
        load_query = {
                'exp_id': loadexpId, 
                'saved_filters': True, 
                'step': args.loadstep}
    load_params = {
            'host': 'localhost',
            'port': loadport,
            'dbname': args.loaddbname,
            'collname': args.loadcolname,
            'exp_id': loadexpId,
            'do_restore': True,
            'query': load_query,
            'from_ckpt': args.ckpt_file}
    return load_params


def get_model_params_from_arg(args):
    model_params = {
            'func': net_builder.get_network_outputs,
            'model_type': args.model_type,
            'prep_type': args.prep_type,
            'setting_name': args.setting_name}
    return model_params


def online_agg_append(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(v)
    return agg_res


def agg_apply_bench(res, benchmark):
    final_res = {}
    assert 'noise_ceiling' not in res, "Invalid key found in res!"
    for layer_name, layer_resp in res.items():
        curr_data = np.concatenate(layer_resp, axis=0)
        model_wrapper = build_wrapper(curr_data, benchmark._assembly)
        score = benchmark(model_wrapper)
        final_res['noise_ceiling'] = np.asarray(score.ceiling.raw)
        final_res[layer_name] = np.asarray(score.raw.raw)
        print(layer_name, np.asarray(score))
    return final_res


def agg_apply_pls_bench(res, benchmark, pca_path):
    final_res = {}
    assert 'noise_ceiling' not in res, "Invalid key found in res!"
    assert os.path.isfile(pca_path), 'PCA file not exist!'
    all_pcas = pickle.load(open(pca_path, 'rb'))
    for layer_name, layer_resp in res.items():
        curr_data = np.concatenate(layer_resp, axis=0)
        curr_data = curr_data.reshape((curr_data.shape[0], -1))
        curr_data = all_pcas[layer_name].transform(curr_data)
        model_wrapper = build_pls_wrapper(curr_data, benchmark._assembly)
        score = benchmark(model_wrapper)
        final_res['noise_ceiling'] = np.asarray(score.ceiling.raw)
        final_res[layer_name] = np.asarray(score.raw.raw)
        print(layer_name, np.asarray(score))
    return final_res


def val_target_func(
        inputs, 
        output, 
        **kwargs):
    return output


def get_validation_params_from_arg(args):
    benchmark = bench.load(args.benchmark)
    assembly = benchmark._assembly
    stimulus_set = assembly.stimulus_set
    folder_dir = os.path.dirname(
            stimulus_set.get_image(stimulus_set['image_id'].values[0]))
    image_list = np.asarray(assembly.image_file_name)
    val_data_param = {
            'func': dataset_func,
            'batch_size': args.batchsize,
            'data_norm_type': args.data_norm_type,
            'img_out_size': args.img_out_size,
            'folder_dir': folder_dir,
            'image_list': image_list}
    benchmark_param = {
            'data_params': val_data_param,
            'targets': {'func': val_target_func},
            'num_steps': int(np.ceil(len(image_list) * 1.0 / args.batchsize)),
            'online_agg_func': online_agg_append,
            'agg_func': lambda res: agg_apply_bench(res, benchmark)}
    if 'pls' in args.benchmark:
        pca_name = '{db_name}_{col_name}_{exp_name}'.format(
                db_name=args.loaddbname,
                col_name=args.loadcolname,
                exp_name=args.loadexpId)
        pca_path = os.path.join(
                '/mnt/fs4/chengxuz/v4it_temp_results/pca_results',
                pca_name + '.pkl')
        benchmark_param['agg_func'] \
                = lambda res: agg_apply_pls_bench(res, benchmark, pca_path)
    validation_params = {'benchmark': benchmark_param}
    return validation_params


def get_params_from_arg(args):
    assert args.benchmark, 'Must specify benchmark'
    save_params = get_save_params_from_arg(args)
    load_params = get_load_params_from_arg(args)
    model_params = get_model_params_from_arg(args)
    validation_params = get_validation_params_from_arg(args)
    params = {
        'save_params': save_params,
        'load_params': load_params,
        'model_params': model_params,
        'validation_params': validation_params,
        'skip_check': True}
    return params


def main():
    parser = get_brainscore_parser()
    args = parser.parse_args()
    args = load_setting(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = get_params_from_arg(args)
    base.test_from_params(**params)


if __name__ == '__main__':
    main()
