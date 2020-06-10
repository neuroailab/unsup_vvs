import argparse
import copy
import tensorflow as tf
import torch
import torch.backends.cudnn as cudnn
import pdb
from tqdm import tqdm
import os
import sys
import json
import numpy as np
import pickle
from model_tools.activations.pytorch import PytorchWrapper
sys.path.append('../combine_pred/')
import cmd_parser
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./brainscore_mask'))
from cleaned_network_builder import get_network_outputs
import bs_fit_neural as bs_fit
import circular_data_loader
import circular_utils

V1_OUT_SHAPE = 166
FIT_CKPT_PATH_PATTERN = os.path.join(
        '/mnt/fs4/chengxuz/v4it_temp_results/.result_caching',
        'layer_param_run_ckpts',
        'model_id={model_id}{id_suffix}-layer-param-run-{layer},bench_id={bench}',
        'split_{which_split}',
        'model.ckpt',
        )
RESULT_PATH_PATTERN = os.path.join(
        '/mnt/fs4/chengxuz/v4it_temp_results/.result_caching',
        'circular_variance',
        'model_id={model_id}{id_suffix},bench_id={bench}',
        'split_{which_split}',
        '{layer}.pkl',
        )


def get_circular_var_parser():
    parser = argparse.ArgumentParser(
            description='The script to compute circular variance')
    parser.add_argument(
            '--gpu', default='0', type=str, action='store')
    parser.add_argument(
            '--id_suffix', default='saver', type=str, action='store')
    parser = bs_fit.add_load_settings(parser)
    parser = bs_fit.add_model_settings(parser)
    return parser


class CircularVarCompute(object):
    def __init__(self, args):
        self.args = args
        self.build_model()
        self.which_split = 0
        self.id_suffix = '-' + args.id_suffix
        self.bench = 'tolias.Cadena2017-param-mask'
        self.num_steps = 5

    def _build_model_ending_points(self):
        args = self.args
        self.data_iter = circular_data_loader.get_iter()
        imgs = self.data_iter['images']
        self.ending_points, _ = get_network_outputs(
                {'images': imgs},
                prep_type=args.prep_type,
                model_type=args.model_type,
                setting_name=args.setting_name,
                module_name=['encode'],
                **json.loads(args.cfg_kwargs))

    def _restore_model_weights(self):
        args = self.args
        SESS = bs_fit.get_tf_sess_restore_model_weight(args)
        self.SESS = SESS
        if getattr(args, 'identifier', None) is None:
            self.model_id = '-'.join(
                    [args.load_dbname,
                     args.load_colname,
                     args.load_expId,
                     str(args.load_port),
                     str(args.load_step)]
                    )
        else:
            self.model_id = args.identifier

    def build_tf_model(self):
        self._build_model_ending_points()
        self._restore_model_weights()
        self.layers = bs_fit.TF_RES18_LAYERS

    def build_dc_model(self):
        args = self.args
        self.SESS = bs_fit.get_tf_sess()
        assert getattr(args, 'load_from_ckpt', None) is not None, \
                "Must specify ckpt to load from"
        self.data_iter = circular_data_loader.get_iter()
        self.data_iter['images'] = tf.transpose(
                circular_data_loader.color_normalize(self.data_iter['images']),
                [0, 3, 1, 2])
        dc_model, self.dc_do_sobel = bs_fit.get_dc_model_do_sobel(
                args.load_from_ckpt)
        self.pt_model = PytorchWrapper(model=dc_model, preprocessing=None)
        self.layers = bs_fit.PT_RES18_LAYERS
        self.model_id = args.identifier

    def build_model(self):
        pt_model_type = getattr(self.args, 'pt_model', None)
        if pt_model_type is None:
            return self.build_tf_model()
        if pt_model_type == 'deepcluster':
            return self.build_dc_model()
        raise NotImplementedError

    def _build_mask_predictor(self, out_shape = V1_OUT_SHAPE):
        with tf.variable_scope('mapping', reuse=tf.AUTO_REUSE):
            input_shape = self._input.shape
            _, spa_x_shape, spa_y_shape, depth_shape = input_shape

            s_w_shape = (spa_x_shape, spa_y_shape, 1, out_shape)
            s_w = tf.get_variable(
                    name='spatial_mask',
                    initializer=tf.contrib.layers.xavier_initializer(),
                    shape=s_w_shape,
                    dtype=tf.float32)

            d_w_shape = (1, 1, depth_shape, out_shape)
            d_w = tf.get_variable(
                    name='depth_mask',
                    initializer=tf.contrib.layers.xavier_initializer(),
                    shape=d_w_shape,
                    dtype=tf.float32)

            bias = tf.get_variable(
                    name='bias',
                    initializer=tf.contrib.layers.xavier_initializer(),
                    shape=out_shape,
                    dtype=tf.float32)

            kernel = s_w * d_w
            kernel = tf.reshape(kernel, [-1, out_shape])
            inputs = tf.layers.flatten(self._input)
            self._predictions = tf.matmul(inputs, kernel)
            self._predictions = tf.nn.bias_add(self._predictions, bias)

        mask_ckpt_path = FIT_CKPT_PATH_PATTERN.format(
                model_id = self.model_id,
                id_suffix = self.id_suffix,
                layer = self.layer,
                bench = self.bench,
                which_split = self.which_split,
                )
        saver = tf.train.Saver(
                var_list={
                    'mapping/spatial_mask': s_w, 
                    'mapping/depth_mask': d_w, 
                    'mapping/bias': bias})
        saver.restore(self.SESS, mask_ckpt_path)
        assert len(self.SESS.run(tf.report_uninitialized_variables())) == 0, \
                (self.SESS.run(tf.report_uninitialized_variables()))

    def __get_tf_model_responses_labels(self):
        layer = self.layer
        assert layer in self.ending_points
        self._input = self.ending_points[layer]
        with tf.variable_scope(layer, reuse=tf.AUTO_REUSE):
            self._build_mask_predictor()

        all_resps = []
        all_labels = []
        for _ in range(self.num_steps):
            _resp, _label = self.SESS.run(
                    [self._predictions, self.data_iter['labels']])
            all_resps.append(_resp)
            all_labels.append(_label)

        flat_features = np.concatenate(all_resps, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return flat_features, labels

    def __get_dc_model_responses_labels(self):
        all_resps = []
        all_labels = []
        for step_idx in range(self.num_steps):
            _imgs, _label = self.SESS.run(
                    [self.data_iter['images'], self.data_iter['labels']])
            _imgs = self.dc_do_sobel(_imgs)
            layer_resp = self.pt_model.get_activations(_imgs, [self.layer])
            layer_resp = np.transpose(layer_resp[self.layer], [0, 2, 3, 1])
            if step_idx == 0:
                self._input = tf.placeholder(
                        dtype=tf.float32, shape=layer_resp.shape)
                with tf.variable_scope(self.layer, reuse=tf.AUTO_REUSE):
                    self._build_mask_predictor()
            _resp = self.SESS.run(
                    self._predictions, feed_dict={self._input: layer_resp})

            all_resps.append(_resp)
            all_labels.append(_label)

        flat_features = np.concatenate(all_resps, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return flat_features, labels

    def get_layer_simulated_neuron_labels(self):
        pt_model_type = getattr(self.args, 'pt_model', None)
        if pt_model_type is None:
            return self.__get_tf_model_responses_labels()
        if pt_model_type == 'deepcluster':
            return self.__get_dc_model_responses_labels()
        raise NotImplementedError

    def recover_flat_features(self, flat_features):
        v1_cadena_data_statistics = pickle.load(
                open('/mnt/fs4/chengxuz/v4it_temp_results/v1_cadena_vars.pkl', 
                    'rb'))
        v1_vars = v1_cadena_data_statistics['cln_var']
        v1_means = v1_cadena_data_statistics['neuron_means']
        flat_features = (flat_features + v1_means[np.newaxis, :]) \
                        * v1_vars[np.newaxis, :]
        flat_features = np.maximum(flat_features, 0)
        return flat_features

    def get_tuning_curves(self, flat_features, labels):
        param_names = ["angles", "sfs", "phases", "colors"]
        tuning_curves = circular_utils.compute_tuning_curves(
                labels,
                flat_features.T,
                passing_indices=None,
                agg_func=np.mean,
                verbose=False,
                param_names=param_names,
                )
        return tuning_curves

    def compute_save_cir_var(self, layer):
        self.layer = layer
        flat_features, labels = self.get_layer_simulated_neuron_labels()
        flat_features = self.recover_flat_features(flat_features)
        tuning_curves = self.get_tuning_curves(flat_features, labels)

        result_path = RESULT_PATH_PATTERN.format(
                model_id = self.model_id,
                id_suffix = self.id_suffix,
                layer = self.layer,
                bench = self.bench,
                which_split = self.which_split,
                )
        save_dir = os.path.dirname(result_path)
        if not os.path.isdir(save_dir):
            os.system('mkdir -p ' + save_dir)
        save_result = {
                'tuning_curves': tuning_curves,
                'flat_features': flat_features,
                'labels': labels,
                }
        pickle.dump(save_result, open(result_path, 'wb'))


def main():
    parser = get_circular_var_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.set_func, "Must specify set_func"
    args = bs_fit.load_set_func(args)

    cir_var_compute = CircularVarCompute(args)
    layers = cir_var_compute.layers
    for which_split in range(4):
        cir_var_compute.which_split = which_split
        for layer in tqdm(layers):
            cir_var_compute.compute_save_cir_var(layer)


if __name__ == '__main__':
    main()
