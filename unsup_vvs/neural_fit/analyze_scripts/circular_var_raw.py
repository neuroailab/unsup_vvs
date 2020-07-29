from tqdm import tqdm
import os
import sys
import numpy as np
import pickle
import tensorflow as tf
import pdb
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./brainscore_mask'))
import circular_var
import bs_fit_neural as bs_fit
RESULT_PATH_PATTERN = os.path.join(
        '/mnt/fs4/chengxuz/v4it_temp_results/.result_caching',
        'circular_variance',
        'model_id={model_id}{id_suffix}',
        'split_{which_split}',
        '{layer}_raw.pkl',
        )


class RawCircularVarCompute(circular_var.CircularVarCompute):
    def __init__(self, args):
        self.args = args
        self.build_model()
        self.which_split = 0
        self.id_suffix = '-' + args.id_suffix
        self.num_steps = 5

    def __get_tf_model_responses_labels(self):
        layer = self.layer
        assert layer in self.ending_points
        self._predictions = self.ending_points[layer]
        self._predictions = tf.layers.flatten(self._predictions)

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
            _resp = layer_resp[self.layer]
            _resp = _resp.reshape([_resp.shape[0], -1])

            all_resps.append(_resp)
            all_labels.append(_label)

        flat_features = np.concatenate(all_resps, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return flat_features, labels

    def get_layer_features_labels(self):
        pt_model_type = getattr(self.args, 'pt_model', None)
        if pt_model_type is None:
            return self.__get_tf_model_responses_labels()
        if pt_model_type == 'deepcluster':
            return self.__get_dc_model_responses_labels()
        raise NotImplementedError

    def compute_save_cir_var(self, layer):
        self.layer = layer
        flat_features, labels = self.get_layer_features_labels()
        tuning_curves = self.get_tuning_curves(flat_features, labels)

        result_path = RESULT_PATH_PATTERN.format(
                model_id = self.model_id,
                id_suffix = self.id_suffix,
                layer = self.layer,
                which_split = self.which_split,
                )
        save_dir = os.path.dirname(result_path)
        if not os.path.isdir(save_dir):
            os.system('mkdir -p ' + save_dir)
        save_result = {
                'tuning_curves': tuning_curves,
                'labels': labels,
                }
        pickle.dump(save_result, open(result_path, 'wb'))


def main():
    parser = circular_var.get_circular_var_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.set_func, "Must specify set_func"
    args = bs_fit.load_set_func(args)

    cir_var_compute = RawCircularVarCompute(args)
    layers = cir_var_compute.layers
    #for which_split in range(4):
    for which_split in range(1):
        cir_var_compute.which_split = which_split
        for layer in tqdm(layers):
            cir_var_compute.compute_save_cir_var(layer)


if __name__ == '__main__':
    main()
