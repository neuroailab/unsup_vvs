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
from brainscore_mask import tf_model_loader
import bs_fit_neural as bs_fit
import circular_var
REGION_TO_BENCH_MAP = {
        'V1': 'tolias.Cadena2017-param-mask',
        'V4': 'dicarlo.Majaj2015.V4-param_mask',
        'IT': 'dicarlo.Majaj2015.IT-param_mask',
        }
REGION_TO_OUTSHAPE_MAP = {
        'V1': 166,
        'V4': 88,
        'IT': 168,
        }
REGION_TO_INPUT_RESOLUTION_MAP = {
        'V1': 40,
        'V4': 224,
        'IT': 224,
        }
NUM_STEPS = 600
RESULT_PATH_PATTERN = os.path.join(
        '/mnt/fs4/chengxuz/v4it_temp_results/.result_caching',
        'optimal_stimuli',
        'model_id={model_id}{id_suffix},bench_id={bench}',
        'split_{which_split}',
        '{layer}{special}.pkl',
        )
NUM_SPLITS = 2


def add_stimuli_settings(parser):
    parser.add_argument(
            '--gpu', default='0', type=str, action='store')
    parser.add_argument(
            '--id_suffix', default='saver', type=str, action='store')
    parser.add_argument(
            '--region', default='V4', type=str, action='store')
    parser.add_argument(
            '--lr', default=0.05, type=float, action='store')
    parser.add_argument(
            '--wd', default=1e-4, type=float, action='store')
    parser.add_argument(
            '--with_xforms', action='store_true')
    parser.add_argument(
            '--just_raw', action='store_true')
    parser.add_argument(
            '--with_tv_jitter', action='store_true')
    parser.add_argument(
            '--tv_wd', default=1e-4, type=float, action='store')
    parser.add_argument(
            '--batch_size', default=32, type=int, action='store',
            help='Batch size')
    parser.add_argument(
            '--layers', type=str, 
            default=None,
            action='store')
    parser.add_argument(
            '--special', default=None, type=str, action='store')
    return parser


def get_stimuli_compute_parser():
    parser = argparse.ArgumentParser(
            description='The script to compute the optimal stimuli')
    parser = add_stimuli_settings(parser)
    parser = bs_fit.add_load_settings(parser)
    parser = bs_fit.add_model_settings(parser)
    return parser


def get_tf_sess_restore_model_weight(args):
    all_vars = tf.global_variables()
    var_list = [x for x in all_vars if x.name.startswith('encode')]
    saver = tf.train.Saver(var_list=var_list)
    SESS = bs_fit.get_tf_sess()

    init_op_global = tf.global_variables_initializer()
    SESS.run(init_op_global)
    init_op_local = tf.local_variables_initializer()
    SESS.run(init_op_local)

    if not args.from_scratch:
        if not args.load_from_ckpt:
            model_ckpt_path = tf_model_loader.load_model_from_mgdb(
                    db=args.load_dbname,
                    col=args.load_colname,
                    exp=args.load_expId,
                    port=args.load_port,
                    cache_dir=args.model_cache_dir,
                    step_num=args.load_step,
                    )
        else:
            model_ckpt_path = args.load_from_ckpt
        saver.restore(SESS, model_ckpt_path)
    return SESS


class StimuliCompute(circular_var.CircularVarCompute):
    def __init__(self, args, start_idx=0, batch_size=None):
        self.args = args
        self.which_split = 0
        self.id_suffix = '-' + args.id_suffix
        self.bench = REGION_TO_BENCH_MAP[args.region]
        self.start_idx = start_idx
        self.batch_size = batch_size
        self.build_model()

    def build_input_images(self):
        args = self.args
        image_initializer = tf.random_uniform_initializer(-0.5, 0.5)
        image_regularizer = tf.contrib.layers.l2_regularizer(args.wd)

        input_resolution = REGION_TO_INPUT_RESOLUTION_MAP[args.region]
        num_of_images = REGION_TO_OUTSHAPE_MAP[args.region]
        if args.just_raw:
            num_of_images = 64
        if self.batch_size is not None:
            num_of_images = self.batch_size
        image_shape = (
                num_of_images,
                input_resolution, input_resolution, 
                1)
        images = tf.get_variable(
                "images",
                image_shape,
                initializer=image_initializer,
                regularizer=image_regularizer)
        self.images_var = images

        if args.with_xforms:
            scales = [1 + (i - 5) / 50. for i in range(11)]
            angles = list(range(-10, 11)) + 5 * [0]
            images = xforms.pad(images, pad_amount=12)
            images = xforms.jitter(images, jitter_amount=8)
            images = xforms.random_scale(images, scales)
            images = xforms.random_rotate(images, angles)
            images = xforms.jitter(images, jitter_amount=4)
        if args.with_tv_jitter:
            import xforms
            images = xforms.jitter(images, jitter_amount=10)
            images = xforms.pad(images, pad_amount=5)
        images = (images + 0.5) * 255
        images = tf.clip_by_value(images, 0, 255)
        images = tf.tile(images, [1, 1, 1, 3])
        self.images = images

    def _build_model_ending_points(self):
        args = self.args
        self.build_input_images()
        self.ending_points, _ = get_network_outputs(
                {'images': self.images},
                prep_type=args.prep_type,
                model_type=args.model_type,
                setting_name=args.setting_name,
                module_name=['encode'],
                **json.loads(args.cfg_kwargs))

    def _restore_model_weights(self):
        args = self.args
        SESS = get_tf_sess_restore_model_weight(args)
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

    def build_model(self):
        pt_model_type = getattr(self.args, 'pt_model', None)
        if pt_model_type is None:
            return self.build_tf_model()
        raise NotImplementedError

    def build_tf_predictions(self):
        layer = self.layer
        assert layer in self.ending_points
        self._input = self.ending_points[layer]
        if not self.args.just_raw:
            with tf.variable_scope(layer, reuse=tf.AUTO_REUSE):
                self._build_mask_predictor(
                        out_shape=REGION_TO_OUTSHAPE_MAP[self.args.region])
                if self.batch_size is not None:
                    self._predictions = self._predictions[:, self.start_idx : self.start_idx + self.batch_size]
        else:
            input_shape = self._input.get_shape().as_list()[1]
            self._predictions = self._input[
                    :, int(input_shape / 2), int(input_shape / 2), :]

    def set_vars(self):
        # remove all other parameters
        all_train_ref = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)

        def _remove_others(vars_ref):
            cp_vars_ref = copy.copy(vars_ref)
            for each_v in cp_vars_ref:
                if each_v.op.name.startswith('images'):
                    continue
                else:
                    vars_ref.remove(each_v)
        _remove_others(all_train_ref)

        all_reg_ref = tf.get_collection_ref(tf.GraphKeys.REGULARIZATION_LOSSES)
        _remove_others(all_reg_ref)

    def get_total_var_loss(self):
        total_variation = \
                tf.reduce_sum(
                        tf.abs(self.images_var[:, :-1, :, :] \
                               - self.images_var[:, 1:, :, :])) \
                + tf.reduce_sum(
                        tf.abs(self.images_var[:, :, :-1, :] \
                                - self.images_var[:, :, 1:, :]))
        return total_variation

    def build_loss(self):
        self.build_tf_predictions()
        self.set_vars()
        loss = tf.linalg.trace(self._predictions)
        self.loss = tf.negative(loss) + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        if self.args.with_tv_jitter:
            total_variation = self.get_total_var_loss()
            self.loss += self.args.tv_wd * total_variation

    def build_train_op(self):
        all_vars = tf.global_variables()
        optimizer = tf.train.AdamOptimizer(self.args.lr)
        self.train_op = optimizer.minimize(self.loss)
        new_all_vars = tf.global_variables()
        added_vars = [x for x in new_all_vars if x not in all_vars]
        added_vars.append(self.images_var)
        init_new_vars_op = tf.initialize_variables(added_vars)
        self.SESS.run(init_new_vars_op)

    def train(self):
        loss_vals = []
        for _ in range(NUM_STEPS):
            _, loss_val = self.SESS.run([self.train_op, self.loss])
            loss_vals.append(loss_val)
        return loss_vals

    def compute_optimal_stimuli(self, layer):
        self.layer = layer
        with tf.variable_scope(
                'split_%i.%s' % (self.which_split, layer)):
            self.build_loss()
            self.build_train_op()
        loss_vals = self.train()
        save_result = {
                'images': self.SESS.run(self.images),
                'loss_vals': loss_vals,
                'actual_response': self.SESS.run(
                    tf.linalg.tensor_diag_part(self._predictions))
                }
        return save_result

    def dump_result(self, save_result):
        special = ''
        if self.args.just_raw:
            special = '_raw'
        if self.args.with_tv_jitter:
            special = '_tv_jitter'
        special = self.args.special or special
        result_path = RESULT_PATH_PATTERN.format(
                model_id = self.model_id,
                id_suffix = self.id_suffix,
                layer = self.layer,
                bench = self.bench,
                which_split = self.which_split,
                special = special,
                )
        save_dir = os.path.dirname(result_path)
        if not os.path.isdir(save_dir):
            os.system('mkdir -p ' + save_dir)
        pickle.dump(save_result, open(result_path, 'wb'))

    def close(self):
        """
        Closes occupied resources
        """
        import tensorflow as tf
        tf.reset_default_graph()
        self.SESS.close()


def aggregate_results(list_of_all_results):
    final_result = {}
    layers = list_of_all_results[0][0].keys()
    for which_split in range(NUM_SPLITS):
        if which_split not in final_result:
            final_result[which_split] = {}
        for layer in layers:
            images = []
            loss_vals = []
            actual_response = []
            for each_all_results in list_of_all_results:
                images.append(each_all_results[which_split][layer]['images'])
                loss_vals.append(each_all_results[which_split][layer]['loss_vals'])
                actual_response.append(each_all_results[which_split][layer]['actual_response'])
            images = np.concatenate(images, axis=0)
            loss_vals = np.sum(loss_vals, axis=0)
            actual_response = np.concatenate(actual_response, axis=0)
            final_result[which_split][layer] = {
                    'images': images,
                    'loss_vals': loss_vals,
                    'actual_response': actual_response,
                    }
    return final_result


def compute_aggregate_dump(args, builder):
    list_of_all_results = []
    num_neurons = REGION_TO_OUTSHAPE_MAP[args.region]
    for start_idx in range(0, num_neurons, args.batch_size):
        all_results = {}
        batch_size = min(args.batch_size, num_neurons - start_idx)
        stimuli_compute = builder(
                args, start_idx=start_idx, 
                batch_size=batch_size)
        layers = stimuli_compute.layers
        if args.just_raw:
            layers = layers[:2]
        if args.layers is not None:
            layers = args.layers.split(',')

        for which_split in range(NUM_SPLITS):
            if which_split not in all_results:
                all_results[which_split] = {}

            stimuli_compute.which_split = which_split
            for layer in tqdm(layers):
                all_results[which_split][layer] = stimuli_compute.compute_optimal_stimuli(layer)

        stimuli_compute.close()
        list_of_all_results.append(all_results)

    final_result = aggregate_results(list_of_all_results)
    for which_split in range(NUM_SPLITS):
        for layer in final_result[which_split]:
            stimuli_compute.layer = layer
            stimuli_compute.which_split = which_split
            stimuli_compute.dump_result(final_result[which_split][layer])


def main():
    parser = get_stimuli_compute_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.set_func, "Must specify set_func"
    args = bs_fit.load_set_func(args)

    compute_aggregate_dump(args, builder=StimuliCompute)


if __name__ == '__main__':
    main()
