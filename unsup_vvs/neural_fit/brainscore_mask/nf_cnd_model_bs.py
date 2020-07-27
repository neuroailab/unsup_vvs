from model_tools.activations.tensorflow import load_resize_image
import tensorflow as tf
from cleaned_network_builder import get_network_outputs
import os
import pdb
import numpy as np
import json
from model_tools.activations.tensorflow import TensorflowSlimWrapper
from candidate_models import score_model
from model_tools.brain_transformation import LayerScores
from model_tools.activations.pca import LayerPCA
import brainscore.benchmarks as bench
from brainscore_mask.param_search_neural import LayerParamScores, ParamScores
from brainscore_mask.majaj2015_mask import DicarloMajaj2015ITLowMidVarMask
from brainscore_mask.majaj2015_mask import DicarloMajaj2015ITMaskParams


def get_imgnt_dir_cats():
    imagenet_raw_dir = '/data5/chengxuz/Dataset/imagenet_raw/'
    val_dir = os.path.join(imagenet_raw_dir, 'val')
    all_cats = os.listdir(val_dir)
    all_cats.sort()
    return val_dir, all_cats


def build_model_ending_points(img_paths):
    image_size = 224
    _load_func = lambda image_path: load_resize_image(image_path, image_size)
    imgs = tf.map_fn(_load_func, img_paths, dtype=tf.float32)

    setting_name = 'cate_res18_exp1'
    ending_points, _ = get_network_outputs(
            {'images': imgs},
            prep_type='mean_std',
            model_type='vm_model',
            setting_name=setting_name,
            #module_name=['encode', 'category'],
            module_name=['encode'],
            )
    for key in ending_points:
        ending_points[key] = tf.transpose(ending_points[key], [0, 3, 1, 2])
    return ending_points


def _get_cate_seed0_ending_points_sess(
        batch_size=50,
        model_ckpt_path='/mnt/fs4/chengxuz/brainscore_model_caches/cate_aug/res18/exp_seed0/checkpoint-505505',
        ):
    img_path_placeholder = tf.placeholder(dtype=tf.string, shape=[batch_size])
    with tf.device('/gpu:0'):
        ending_points = build_model_ending_points(img_path_placeholder)
    
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    SESS = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options,
            ))
    #model_ckpt_path = '/mnt/fs4/chengxuz/brainscore_model_caches/cate_aug/res18/exp_seed0/checkpoint-505505'
    #model_ckpt_path = '/mnt/fs4/chengxuz/brainscore_model_caches/irla_and_others/res18/la/checkpoint-2502500'
    #model_ckpt_path = '/mnt/fs4/chengxuz/brainscore_model_caches/cate_aug/res18/exp_seed1/checkpoint-505505'
    saver.restore(SESS, model_ckpt_path)
    assert len(SESS.run(tf.report_uninitialized_variables())) == 0, \
            (SESS.run(tf.report_uninitialized_variables()))
    return img_path_placeholder, ending_points, SESS


def test_build_model_ending_points():
    img_path_placeholder, ending_points, SESS = _get_cate_seed0_ending_points_sess()

    val_dir, all_cats = get_imgnt_dir_cats()
    all_test_cats = [0, 1]
    for each_test_cat in all_test_cats:
        _cat_val_dir = os.path.join(val_dir, all_cats[each_test_cat])
        _cat_val_img_paths = [
                os.path.join(_cat_val_dir, _img_path) \
                for _img_path in os.listdir(_cat_val_dir)]
        category_outputs = SESS.run(
                ending_points['category_2'], 
                feed_dict={img_path_placeholder: _cat_val_img_paths})
        pred_category = np.argmax(category_outputs, axis=1)
        print('Accuracy for category %i: %i/50' \
                % (each_test_cat, 
                   np.sum(pred_category == each_test_cat)))


def test_bs_PCA_fit():
    img_path_placeholder, ending_points, SESS \
            = _get_cate_seed0_ending_points_sess(
                    64,
                    '/mnt/fs4/chengxuz/brainscore_model_caches/cate_aug/res18/exp_seed0_bn_wd/checkpoint-505505')
    identifier = 'test-cate-bn-wd-pca'
    activations_model = TensorflowSlimWrapper(
            identifier=identifier, labels_offset=0,
            endpoints=ending_points, inputs=img_path_placeholder, 
            session=SESS)

    LayerPCA.hook(activations_model, n_components=1000)
    #benchmark = bench.load('dicarlo.Majaj2015.IT-pls')
    benchmark = bench.load('dicarlo.Majaj2015.V4-pls')
    all_layers = ['encode_%i' % i for i in range(1, 10)]

    model_scores = LayerScores(
            identifier, 
            activations_model=activations_model,
            )
    score = model_scores(
            benchmark=benchmark, 
            layers=all_layers,
            )
    print(score)


def test_bs_fit():
    img_path_placeholder, ending_points, SESS \
            = _get_cate_seed0_ending_points_sess(64)

    #identifier = 'test-la-seed0'
    #identifier = 'test-la-seed0-2'
    #identifier = 'test-cate-seed1'
    #identifier = 'test-cate-seed0-mask-2'
    #identifier = 'test-cate-seed0-mask-midvar'
    #identifier = 'test-cate-seed0-mask-midvar-smwd'
    #identifier = 'test-cate-seed0-mask-midvar-params'
    #identifier = 'test-cate-seed0-mask-midvar-faster'
    #identifier = 'test-cate-seed0-mask-midvar-faster-debug'
    #identifier = 'test-cate-seed0-mask-midvar-faster-debug-2'
    #identifier = 'test-cate-seed0-mask-final-2'
    #identifier = 'test-cate-seed0-mask-final-e12'
    #identifier = 'test-cate-seed0-mask-final-e1-nolap'
    #identifier = 'test-cate-seed0-mask-final-e1-nolap-3'
    identifier = 'test-cate-seed0-mask-final-all-test-2'
    activations_model = TensorflowSlimWrapper(
            identifier=identifier, labels_offset=0,
            endpoints=ending_points, inputs=img_path_placeholder, 
            session=SESS)

    #all_params = []
    #for each_ls in [0.1, 0.01]:
    #    for each_ld in [0.1, 0.01]:
    #        all_params.append(json.dumps([[], {'ls': each_ls, 'ld': each_ld}]))

    #all_params = [
    #        json.dumps([[], {'ls': 0.1, 'ld': 0.1}]), 
    #        json.dumps([[], {'ls': 0.01, 'ld': 0.01}])]

    all_params = [
            json.dumps([[], {
                'with_corr_loss': True,
                'with_lap_loss': False,
                'adam_eps': 0.1,
                'max_epochs': 50, 
                'decay_rate': 35,
                }]), 
            #json.dumps([[], {'max_epochs': 50, 'decay_rate': 35, 'ls': 0.01, 'ld': 0.01}]), 
            #json.dumps([[], {'max_epochs': 60, 'decay_rate': 45}]), 
            ]

    layer_and_params = []
    #all_layers = ['encode_%i' % i for i in range(1, 10)]
    all_layers = ['encode_1']
    for layer in all_layers:
        _model_scores = ParamScores(
                identifier, 
                activations_model=activations_model)
        _score = _model_scores(
                #benchmark_builder=DicarloMajaj2015ITLowMidVarMask, 
                benchmark_builder=DicarloMajaj2015ITMaskParams, 
                params=all_params,
                layer=layer,
                )
        print(_score)
        #layer_and_params.append(
        #        [layer, 
        #         all_params[int(_score.sel(aggregation='center').argmax())]])

    #model_scores = LayerParamScores(
    #        identifier, 
    #        activations_model=activations_model,
    #        )
    #score = model_scores(
    #        benchmark_builder=DicarloMajaj2015ITMaskParams, 
    #        layer_and_params=layer_and_params,
    #        )
    #print(score)


if __name__ == '__main__':
    #test_build_model_ending_points()
    test_bs_fit()
    #test_bs_PCA_fit()
