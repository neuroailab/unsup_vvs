import os
import sys
import pdb
import copy
import importlib

sys.path.append(os.path.expanduser('~/ntcnet/brainscore_tests/integrated/'))
sys.path.append(os.path.expanduser('~/ntcnet/decoder_search'))
from integrated_graph_model import integrated_graph_convrnn

DEFAULT_MODEL_PARAMS = {
        'model_params_config': 'enet07L_unr0', 
        'unroll_tf':True, 
        'const_pres':True}


def build_integrated_model_params(const_pres=False,
                                  max_internal_time=None,
                                  image_on=0,
                                  image_off=10,
                                  model_params_config=None,
                                  train_targets=[],
                                  unroll_tf=False):
    if const_pres:
        neural_presentation_kwargs = None
    else:
        neural_presentation_kwargs = {
                'image_on': image_on, 
                'image_off':image_off}

    # get model params from a dill
    assert model_params_config is not None
    import dill
    model_config_dir = '~/ntcnet/brainscore_tests/model_configs/'
    pkl_path = model_config_dir + model_params_config + '.pkl'
    with open(os.path.expanduser(pkl_path), 'rb') as f:
        model_params = dill.load(f)
        model_params['base_name'] = os.path.join(
                os.path.expanduser(model_config_dir),
                model_params['base_name'])

    model_func_kwargs = copy.deepcopy(model_params)
    model_func_kwargs.update({
        'train_targets': train_targets,
        'output_global_pool': True,
        'images_key': 'images',
        'unroll_tf': unroll_tf,
        'output_times': None,
        'neural_presentation_kwargs':neural_presentation_kwargs,
        'max_internal_time':max_internal_time,
        'out_layers': ['conv0', 'conv1', 'conv2', 'conv3', 
                       'conv4', 'conv5', 'conv6', 'conv7'],
        'train': False,
    })

    return model_func_kwargs


def convrnn_model_func(
        inputs, 
        builder_kwargs=DEFAULT_MODEL_PARAMS):
    model_func = integrated_graph_convrnn
    model_func_kwargs = build_integrated_model_params(**builder_kwargs)
    outputs = model_func(inputs, **model_func_kwargs)
    return outputs[0]['times'][0]


if __name__ == '__main__':
    import tensorflow as tf
    builder_kwargs = DEFAULT_MODEL_PARAMS
    model_func = integrated_graph_convrnn
    model_func_kwargs = build_integrated_model_params(**builder_kwargs)
    #inputs = {'images': tf.zeros([64, 40, 40, 3], dtype=tf.float32)}
    inputs = {'images': tf.zeros([64, 224, 224, 3], dtype=tf.float32)}
    outputs = model_func(inputs, **model_func_kwargs)
    pdb.set_trace()
