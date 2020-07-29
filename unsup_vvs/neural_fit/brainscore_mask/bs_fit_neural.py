import argparse
import copy
from argparse import Namespace
import tensorflow as tf
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import pickle
import os
import sys
sys.path.append('../network_training/')
from network_training import cmd_parser
sys.path.append(os.path.abspath('./'))
DEFAULT_RESULTCACHING_HOME = '~/.result_caching_unsup_vvs'
os.environ['RESULTCACHING_HOME'] = DEFAULT_RESULTCACHING_HOME
import importlib
import pdb
import numpy as np
import json
from collections import OrderedDict

from model_tools.activations.pytorch import \
        load_images, load_preprocess_images, PytorchWrapper
from model_tools.activations.tensorflow import load_resize_image
from model_tools.activations.tensorflow import TensorflowSlimWrapper
from model_tools.brain_transformation import LayerScores
from model_tools.activations.pca import LayerPCA
import brainscore.benchmarks as bench
from model_tools.brain_transformation import ProbabilitiesMapping

from bs_fit_utils \
        import get_dc_model, load_set_func, get_load_settings_from_func
from bs_fit_utils import color_normalize
import bs_fit_utils as bs_fit_utils
from neural_fit.cleaned_network_builder import get_network_outputs
import tf_model_loader
from param_search_neural import \
        LayerParamScores, ParamScores, LayerActivations, \
        LayerModel, LayerRegressParamScores, RegressParamScores
from majaj2015_mask import \
        DicarloMajaj2015ITLowMidVarMask, DicarloMajaj2015V4LowMidVarMask
from majaj2015_mask import \
        DicarloMajaj2015ITMaskParams, DicarloMajaj2015V4MaskParams
from cadena2017_mask import \
        ToliasCadena2017MaskParams, ToliasCadena2017MaskParamsCadenaScores, \
        ToliasCadena2017WithNaNsMaskParams, ToliasCadena2017CadenaFitCadenaScores, \
        ToliasCadena2017Correlation
from majaj2015_mask import \
        DicarloMajaj2015TransRegLowMidVar, DicarloMajaj2015TransRegHighVar
import behavior_bench as bhv_bench
import majaj2015_mask as majaj2015_bench
DEFAULT_PARAMS = {
        'with_corr_loss': True,
        'with_lap_loss': False,
        'adam_eps': 0.1,
        'max_epochs': 100, 
        'decay_rate': 70,
        'ls': 0.01,
        'ld': 0.01,
        }
PT_RES18_LAYERS = \
        ['relu', 'maxpool'] +\
        ['layer1.0.relu', 'layer1.1.relu'] +\
        ['layer2.0.relu', 'layer2.1.relu'] +\
        ['layer3.0.relu', 'layer3.1.relu'] +\
        ['layer4.0.relu', 'layer4.1.relu']
TF_RES18_LAYERS = ['encode_1.conv'] + ['encode_%i' % i for i in range(1, 10)]


def add_load_settings(parser):
    parser.add_argument(
            '--load_port', default=27009, type=int, action='store',
            help='Port number of mongodb for loading')
    parser.add_argument(
            '--load_expId', default=None,
            type=str, action='store',
            help='Name of experiment id')
    parser.add_argument(
            '--load_step', default=None, type=int, action='store',
            help='Number of steps for loading')
    parser.add_argument(
            '--load_dbname', default=None, type=str, action='store',
            help='Name of experiment database to load from')
    parser.add_argument(
            '--load_colname', default=None, type=str, action='store',
            help='Name of experiment collection name to load from')
    parser.add_argument(
            '--model_cache_dir', 
            default=tf_model_loader.DEFAULT_MODEL_CACHE_DIR, 
            type=str, action='store',
            help='Prefix of cache directory')
    parser.add_argument(
            '--from_scratch',
            action='store_true',
            help='Whether using initializations')
    parser.add_argument(
            '--load_from_ckpt', 
            default=None,
            type=str, action='store',
            help='Ckpt path to load from')
    parser.add_argument(
            '--load_var_list', default=None,
            type=str, action='store',
            help='Var list used for loading')
    parser.add_argument(
            '--set_func', type=str, 
            default=None,
            action='store')
    return parser


def add_brainscore_settings(parser):
    parser.add_argument(
            '--batch_size', default=64, type=int, action='store',
            help='Batch size')
    parser.add_argument(
            '--benchmark', default=None, type=str, action='store',
            help='Benchmark name from brainscore')
    parser.add_argument(
            '--id_suffix', default=None, type=str, action='store',
            help='Suffix of id, mainly for debug purpose')
    parser.add_argument(
            '--identifier', default=None, type=str, action='store',
            help='Identifier, mainly for models from ckpt files')
    return parser


def add_model_settings(parser):
    parser.add_argument(
            '--model_type', default='vm_model', type=str, action='store')
    parser.add_argument(
            '--prep_type', default='mean_std', type=str, action='store')
    parser.add_argument(
            '--setting_name', default=None, type=str, action='store',
            help='Network setting name')
    parser.add_argument(
            '--cfg_kwargs', default="{}", type=str, action='store',
            help='Kwargs for network cfg')
    return parser


def add_general_settings(parser):
    parser.add_argument(
            '--gpu', default='0', type=str, action='store')
    parser.add_argument(
            '--bench_func', type=str, 
            required=True,
            action='store')
    parser.add_argument(
            '--bs_models_id', 
            default=None, type=str, action='store',
            help='If not None, will use this as the model identifier')
    parser.add_argument(
            '--pt_model', 
            default=None, type=str, action='store',
            help='If not None, will use corresponding pytorch models')
    parser.add_argument(
            '--pt_model_mid', 
            default=None, type=str, action='store',
            help='If not None, will add linear mid layer to pytorch models')
    parser.add_argument(
            '--load_prep_type', type=str, 
            default='hvm',
            action='store',
            help='Preprocessing type')
    parser.add_argument(
            '--ls_ld_range', type=str, 
            default='default',
            action='store',
            help='Searching range for ls and ld')
    parser.add_argument(
            '--debug_mode', action='store_true')
    parser.add_argument(
            '--layer_separate', action='store_true')
    parser.add_argument(
            '--param_score_id', 
            default=None, type=str, action='store',
            help='If not None, will use this as the id during param scores')
    parser.add_argument(
            '--just_input', action='store_true')
    parser.add_argument(
            '--special_model', 
            default=None, type=str, action='store',
            help='If not None, will use special model builder')
    parser.add_argument(
            '--layers', type=str, 
            default=None,
            action='store')
    return parser


def get_brainscore_parser():
    parser = argparse.ArgumentParser(
            description='The script to fit to neural data using brain score')
    parser = add_load_settings(parser)
    parser = add_brainscore_settings(parser)
    parser = add_model_settings(parser)
    parser = add_general_settings(parser)
    return parser


def get_tf_sess():
    gpu_options = tf.GPUOptions(allow_growth=True)
    SESS = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options,
            ))
    return SESS


def get_tf_sess_restore_model_weight(args):
    SESS = get_tf_sess()
    if args.load_var_list is not None:
        name_var_list = json.loads(args.load_var_list)
        needed_var_list = {}
        curr_vars = tf.global_variables()
        curr_names = [variable.op.name for variable in curr_vars]
        for old_name in name_var_list:
            new_name = name_var_list[old_name]
            assert new_name in curr_names, "Variable %s not found!" % new_name
            _ts = curr_vars[curr_names.index(new_name)]
            needed_var_list[old_name] = _ts
        saver = tf.train.Saver(needed_var_list)

        init_op_global = tf.global_variables_initializer()
        SESS.run(init_op_global)
        init_op_local = tf.local_variables_initializer()
        SESS.run(init_op_local)
    else:
        saver = tf.train.Saver()

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
    else:
        init_op_global = tf.global_variables_initializer()
        SESS.run(init_op_global)
        init_op_local = tf.local_variables_initializer()
        SESS.run(init_op_local)

    assert len(SESS.run(tf.report_uninitialized_variables())) == 0, \
            (SESS.run(tf.report_uninitialized_variables()))
    return SESS


def get_dc_model_do_sobel(path):
    model = get_dc_model(path)
    sobel_filter = model.sobel
    model = model.features

    def _do_sobel(images):
        device = torch.device("cpu")
        images = torch.autograd.Variable(torch.from_numpy(images).cuda())
        images = sobel_filter(images)
        images = images.float().to(device).numpy()
        return images
    return model, _do_sobel


class ScoreVMModels:
    def __init__(self, args):
        self.args = args
        self.dim_reduce = None
        self.i2n_save_suffix = ''

    def __v1_cadena_load_image(self, image_path):
        image = load_resize_image(image_path, 70)
        image = image[15:55, 15:55]
        return image

    def _get_imgs_from_paths(self, img_paths):
        args = self.args

        if args.load_prep_type == 'hvm':
            _load_func = lambda image_path: load_resize_image(
                    image_path, 224)
        elif args.load_prep_type == 'v1_cadena':
            _load_func = lambda image_path: self.__v1_cadena_load_image(
                    image_path)
        else:
            raise NotImplementedError

        imgs = tf.map_fn(_load_func, img_paths, dtype=tf.float32)
        return imgs

    def _build_model_ending_points(self, img_paths):
        imgs = self._get_imgs_from_paths(img_paths)
        args = self.args

        ending_points, _ = get_network_outputs(
                {'images': imgs},
                prep_type=args.prep_type,
                model_type=args.model_type,
                setting_name=args.setting_name,
                **json.loads(args.cfg_kwargs))
        for key in ending_points:
            if len(ending_points[key].get_shape().as_list()) == 4:
                ending_points[key] = tf.transpose(
                        ending_points[key], 
                        [0, 3, 1, 2])
        if self.dim_reduce is not None:
            if self.dim_reduce == 'spatial_average':
                for key in ending_points:
                    if len(ending_points[key].get_shape().as_list()) == 4:
                        ending_points[key] = tf.reduce_mean(
                                ending_points[key], 
                                axis=(2,3))
            else:
                raise NotImplementedError
        return ending_points

    def _get_brainscore_model(self, id_suffix=None):
        from candidate_models.model_commitments.ml_pool import model_layers_pool
        _id = self.args.bs_models_id
        model_layers = model_layers_pool[_id]
        self.activations_model = model_layers['model']
        self.layers = model_layers['layers']
        self.identifier = 'brainscore-' + _id
        if self.args.id_suffix:
            self.identifier += '-' + self.args.id_suffix
        if id_suffix:
            self.identifier += id_suffix

    def __get_tf_model(self):
        args = self.args
        img_path_placeholder = tf.placeholder(
                dtype=tf.string, 
                shape=[args.batch_size])
        with tf.device('/gpu:0'):
            ending_points = self._build_model_ending_points(
                    img_path_placeholder)
        
        SESS = get_tf_sess_restore_model_weight(args)

        if args.identifier is None:
            identifier = '-'.join(
                    [args.load_dbname,
                     args.load_colname,
                     args.load_expId,
                     str(args.load_port),
                     str(args.load_step)]
                    )
        else:
            identifier = args.identifier
        if args.id_suffix:
            identifier += '-' + args.id_suffix

        self.ending_points = ending_points
        self.img_path_placeholder = img_path_placeholder
        self.SESS = SESS
        self.identifier = identifier
        self.layers = TF_RES18_LAYERS
        if args.layers is not None:
            self.layers = args.layers.split(',')
        if args.debug_mode:
            self.layers = ['encode_1.conv']
        if args.just_input:
            self.layers = ['model_inputs']
        if args.setting_name is not None and args.setting_name.startswith('cate_fc18'):
            self.layers = ['encode_%i' % i for i in range(1, 18, 2)]
        self._build_activations_model()

    def __get_pt_dc_model(self):
        args = self.args
        model, _do_sobel = get_dc_model_do_sobel(args.load_from_ckpt)

        def _hvm_load_preprocess(image_paths):
            images = load_preprocess_images(image_paths, 224)
            return _do_sobel(images)

        def _cadena_load_preprocess(image_paths):
            images = load_preprocess_images(image_paths, 70)
            images = images[:, :, 15:55, 15:55]
            return _do_sobel(images)

        if args.load_prep_type == 'hvm':
            preprocessing = _hvm_load_preprocess
        elif args.load_prep_type == 'v1_cadena':
            preprocessing = _cadena_load_preprocess
        else:
            raise NotImplementedError
        self.model = model
        self.preprocessing = preprocessing
        self.layers = PT_RES18_LAYERS

    def __get_pt_la_cmc_model(self):
        args = self.args
        model = bs_fit_utils.get_la_cmc_model(args.load_from_ckpt)
        self.model = model.module.l_to_ab

        def _do_resize_lab_normalize(images, img_size):
            from pt_scripts.main import tolab_normalize
            from PIL import Image
            post_images = []
            for image in images:
                image = image.resize([img_size, img_size])
                image = np.asarray(image).astype(np.float32)
                if len(image.shape) == 2:
                    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
                image = tolab_normalize(image)
                image = image[:, :, :1]
                image = np.transpose(image, [2, 0, 1])
                post_images.append(image)
            images = np.stack(post_images, axis=0)
            return images

        def _hvm_load_preprocess(image_paths):
            images = load_images(image_paths)
            return _do_resize_lab_normalize(images, 224)

        def _cadena_load_preprocess(image_paths):
            images = load_images(image_paths)
            images = _do_resize_lab_normalize(images, 70)
            images = images[:, :, 15:55, 15:55]
            return images

        if args.load_prep_type == 'hvm':
            preprocessing = _hvm_load_preprocess
        elif args.load_prep_type == 'v1_cadena':
            preprocessing = _cadena_load_preprocess
        else:
            raise NotImplementedError
        self.preprocessing = preprocessing
        self.layers = PT_RES18_LAYERS

    def __get_pt_official_model(self):
        args = self.args

        def _hvm_load_preprocess(image_paths):
            images = load_preprocess_images(image_paths, 224)
            return images

        def _cadena_load_preprocess(image_paths):
            images = load_preprocess_images(image_paths, 70)
            images = images[:, :, 15:55, 15:55]
            return images

        if args.load_prep_type == 'hvm':
            preprocessing = _hvm_load_preprocess
        elif args.load_prep_type == 'v1_cadena':
            preprocessing = _cadena_load_preprocess
        else:
            raise NotImplementedError

        # freeze the layers
        import torchvision.models as models
        model = models.resnet18(pretrained=True)
        model.cuda()
        cudnn.benchmark = True
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self.model = model
        self.preprocessing = preprocessing
        self.layers = PT_RES18_LAYERS

        if args.layers is not None:
            self.layers = args.layers.split(',')

    def __add_pt_mid(self):
        args = self.args
        mid_layer = bs_fit_utils.load_mid_layer(args.pt_model_mid)
        self.model = bs_fit_utils.PtModelWithMid(
                encoder=self.model,
                mid_layer=mid_layer,
                )
        self.layers = ['mid_layer.linear_mid']

    def __get_pt_model(self):
        args = self.args
        self.identifier = args.identifier
        if args.id_suffix:
            self.identifier += '-' + args.id_suffix

        if args.pt_model == 'deepcluster':
            self.__get_pt_dc_model()
        elif args.pt_model in ['la_cmc']:
            self.__get_pt_la_cmc_model()
        elif args.pt_model == 'official':
            self.__get_pt_official_model()
        else:
            raise NotImplementedError
        if args.layers is not None:
            self.layers = args.layers.split(',')
        if args.pt_model_mid is not None:
            self.__add_pt_mid()
        self._build_activations_model()

    def __get_special_model(self):
        # Special vgg-19 from Cadena paper
        args = self.args
        self.identifier = 'cadena-vgg19'
        if args.id_suffix:
            self.identifier += '-' + args.id_suffix

        cadena_repo_path = os.path.expanduser('~/Cadena2019PlosCB')
        sys.path.append(cadena_repo_path)
        from cnn_sys_ident import utils
        args = self.args
        img_path_placeholder = tf.placeholder(
                dtype=tf.string, 
                shape=[args.batch_size])
        with tf.device('/gpu:0'):
            imgs = self._get_imgs_from_paths(img_path_placeholder)
            imgs = color_normalize(imgs)
            ending_points = utils.vgg19(
                    imgs, 
                    subtract_mean=False,
                    padding='SAME',
                    )
        for key in ending_points:
            ending_points[key] = tf.transpose(ending_points[key], [0, 3, 1, 2])
        new_ending_points = OrderedDict()
        for old_key in ending_points:
            if '/' in old_key:
                new_key = old_key.split('/')[-1]
            else:
                new_key = old_key
            new_ending_points[new_key] = ending_points[old_key]
        ending_points = new_ending_points
        
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        SESS = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=gpu_options,
                ))
        model_ckpt_path = os.path.join(
                cadena_repo_path, 
                'vgg_weights/vgg_normalized.ckpt')
        saver.restore(SESS, model_ckpt_path)

        assert len(SESS.run(tf.report_uninitialized_variables())) == 0, \
                (SESS.run(tf.report_uninitialized_variables()))

        self.img_path_placeholder = img_path_placeholder
        self.ending_points = ending_points
        self.SESS = SESS
        self.layers = \
                ['conv1_1', 'conv1_2'] \
                + ['conv2_1', 'conv2_2'] \
                + ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4'] \
                + ['conv4_1', 'conv4_2', 'conv4_3', 'conv4_4'] \
                + ['conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        #self.layers = ['conv1_1']
        self._build_activations_model()

    def _get_activation_model(self):
        args = self.args
        if args.bs_models_id:
            self._get_brainscore_model()
            return
        if args.pt_model:
            self.__get_pt_model()
            return
        if args.special_model:
            self.__get_special_model()
            return
        self.__get_tf_model()

    def _build_activations_model(self, id_suffix=''):
        args = self.args
        self.identifier += id_suffix
        if args.bs_models_id:
            self._get_brainscore_model(id_suffix)
            return
        if args.pt_model:
            self.activations_model = PytorchWrapper(
                    identifier=self.identifier,
                    model=self.model,
                    preprocessing=self.preprocessing,
                    batch_size=args.batch_size,
                    )
            return 
        activations_model = TensorflowSlimWrapper(
                identifier=self.identifier, labels_offset=0,
                endpoints=self.ending_points, 
                inputs=self.img_path_placeholder, 
                session=self.SESS,
                batch_size=args.batch_size)
        self.activations_model = activations_model

    def _run_layer_scores(self):
        model_scores = LayerScores(
                self.identifier, 
                activations_model=self.activations_model,
                visual_degrees=8,
                )
        score = model_scores(
                benchmark=self.benchmark, 
                layers=self.layers,
                )
        return score

    def __add_ckpt_dir_to_layer_and_param(
            self, layer_and_param, model_id, bench_id):
        layer, param_str = layer_and_param
        bench_args, bench_kwargs = json.loads(param_str)
        ckpt_dir = os.path.join(
                DEFAULT_RESULTCACHING_HOME,
                'layer_param_run_ckpts',
                f'model_id={model_id},bench_id={bench_id}'
                )
        bench_kwargs['checkpoint_dir'] = ckpt_dir
        param_str = json.dumps([bench_args, bench_kwargs])
        return [layer, param_str]

    def __get_i2n_save_pkl_path(self, layer):
        model_id = self.identifier
        dir_name = os.path.join(
                DEFAULT_RESULTCACHING_HOME,
                'i2n_matrix_results' + self.i2n_save_suffix,
                f'model_id={model_id}',
                )
        pkl_path = os.path.join(
                dir_name,
                f'{layer}.pkl',
                )
        os.system('mkdir -p ' + dir_name)
        return pkl_path

    def __get_i2n_result_path(self):
        model_id = self.identifier
        result_path = os.path.join(
                DEFAULT_RESULTCACHING_HOME,
                'i2n_results' + self.i2n_save_suffix,
                f'model_id={model_id}.pkl',
                )
        return result_path

    def _run_layer_param_scores(self, benchmark_builder):
        self._build_activations_model('-layer-param-run')
        scores = []
        for layer, layer_and_param \
                in zip(self.layers, self.layer_and_params):
            _tmp_model_id = self.identifier + '-' + layer
            model_scores = LayerParamScores(
                    _tmp_model_id,
                    activations_model=self.activations_model,
                    )
            layer_and_param = self.__add_ckpt_dir_to_layer_and_param(
                    layer_and_param,
                    model_id=_tmp_model_id,
                    bench_id=benchmark_builder().identifier)
            score = model_scores(
                    benchmark_builder=benchmark_builder, 
                    layer_and_params=[layer_and_param],
                    )
            scores.append(score)
        return scores

    def _run_layer_probability_scores(self, benchmark):
        scores = []
        for layer in self.layers:
            _tmp_model_id = self.identifier + '-' + layer
            prob_model = ProbabilitiesMapping(
                    _tmp_model_id,
                    activations_model=self.activations_model,
                    layer=layer,
                    )
            score = benchmark(prob_model)
            scores.append(score)
        return scores

    def _run_layer_probability_scores_with_save(self, benchmark_builder):
        i2n_result_path = self.__get_i2n_result_path()
        if os.path.exists(i2n_result_path):
            scores = pickle.load(open(i2n_result_path, 'rb'))
            return scores
        scores = {}
        for layer in self.layers:
            _tmp_model_id = self.identifier + '-' + layer
            prob_model = ProbabilitiesMapping(
                    _tmp_model_id,
                    activations_model=self.activations_model,
                    layer=layer,
                    )
            pkl_path = self.__get_i2n_save_pkl_path(layer)
            benchmark = benchmark_builder(pkl_path=pkl_path)
            score = benchmark(prob_model)
            scores[layer] = score
        pickle.dump(scores, open(i2n_result_path, 'wb'))
        return scores

    def _run_layer_regression_scores(self, benchmark):
        scores = []
        for layer in self.layers:
            _tmp_model_id = self.identifier + '-' + layer
            layer_model = LayerModel(
                    _tmp_model_id,
                    activations_model=self.activations_model,
                    layers=[layer],
                    )
            score = benchmark(layer_model)
            scores.append(score)
        return scores

    def _run_layer_regression_param_scores(self, benchmark_builder):
        scores = []
        for layer, layer_and_param \
                in zip(self.layers, self.layer_and_params):
            _tmp_model_id = self.identifier + '-' + layer
            model_scores = LayerRegressParamScores(
                    _tmp_model_id,
                    activations_model=self.activations_model)
            score = model_scores(
                    benchmark_builder=benchmark_builder, 
                    layer_and_params=[layer_and_param])
            scores.append(score)
        return scores

    def hvm_var6_activations(self):
        self._get_activation_model()
        self._build_activations_model('-activations')
        self.benchmark = bench.load('dicarlo.Majaj2015.IT-pls')
        layer_activations = LayerActivations(
                self.identifier, 
                activations_model=self.activations_model,
                )
        layer_activations(benchmark=self.benchmark, layers=self.layers)

    def hvm_it_pls_scores(self):
        self._get_activation_model()
        LayerPCA.hook(self.activations_model, n_components=1000)
        self.benchmark = bench.load('dicarlo.Majaj2015.IT-pls')
        score = self._run_layer_scores()
        print(score)

    def hvm_v4_pls_scores(self):
        self._get_activation_model()
        LayerPCA.hook(self.activations_model, n_components=1000)
        self.benchmark = bench.load('dicarlo.Majaj2015.V4-pls')
        score = self._run_layer_scores()
        print(score)

    def freeman_v1_pls_scores(self):
        self._get_activation_model()
        LayerPCA.hook(self.activations_model, n_components=1000)
        self.benchmark = bench.load('movshon.FreemanZiemba2013.V1-pls')
        score = self._run_layer_scores()
        print(score)

    def freeman_v2_pls_scores(self):
        self._get_activation_model()
        LayerPCA.hook(self.activations_model, n_components=1000)
        self.benchmark = bench.load('movshon.FreemanZiemba2013.V2-pls')
        score = self._run_layer_scores()
        print(score)

    def hvm_it_mask_scores(self):
        self._get_activation_model()
        self.benchmark = bench.load('dicarlo.Majaj2015.IT-mask')
        score = self._run_layer_scores()
        print(score)

    def hvm_v4_mask_scores(self):
        self._get_activation_model()
        self.benchmark = bench.load('dicarlo.Majaj2015.V4-mask')
        score = self._run_layer_scores()
        print(score)

    def _get_all_mask_params(self):
        global DEFAULT_PARAMS
        all_params = []
        _ls_ld_list = [
                [1, 1], 
                [0.5, 0.5], 
                [0.1, 0.1], 
                [0.05, 0.05], 
                [0.01, 0.01], 
                [0.005, 0.005], 
                [0.001, 0.001]]
        if self.args.ls_ld_range == 'v1_cadena_range':
            _ls_ld_list = [
                    [0.05, 0.05], 
                    [0.01, 0.01], 
                    [0.005, 0.005], 
                    [0.001, 0.001],
                    [5e-4, 5e-4],
                    [1e-4, 1e-4],
                    [5e-5, 5e-5]]
            DEFAULT_PARAMS['normalize_Y'] = True
        if self.args.ls_ld_range == 'v1_cadena_input_range':
            _ls_ld_list = [
                    [5, 5],
                    [1, 1],
                    [0.5, 0.5],
                    [0.1, 0.1],
                    [0.05, 0.05], 
                    [0.01, 0.01], 
                    ]
            DEFAULT_PARAMS['normalize_Y'] = True

        if self.args.debug_mode:
            DEFAULT_PARAMS['batch_size'] = 256

        for each_ls, each_ld in _ls_ld_list:
            ls_ld_params = {
                    'ls': each_ls,
                    'ld': each_ld,
                    }
            now_params = copy.deepcopy(DEFAULT_PARAMS)
            now_params.update(ls_ld_params)
            all_params.append(json.dumps([[], now_params]))
        self.all_params = all_params

    def _get_layer_and_params(self, benchmark_builder):
        layer_and_params = []
        for layer in tqdm(self.layers, desc='Param layers'):
            _model_scores = ParamScores(
                    self.args.param_score_id or self.identifier, 
                    activations_model=self.activations_model)
            _score = _model_scores(
                    benchmark_builder=benchmark_builder, 
                    params=self.all_params,
                    layer=layer,
                    )
            layer_and_params.append(
                    [layer, 
                     self.all_params[
                         int(_score.sel(aggregation='center').argmax())]])
        self.layer_and_params = layer_and_params

    def _get_all_regression_params(self):
        all_params = []
        c_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        for _c in c_list:
            all_params.append(json.dumps([[], {'C': _c}]))
        self.all_params = all_params

    def _get_all_categorization_params(self):
        all_params = []
        #c_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
        c_list = [1e-6, 1e-5, 1e-4]
        for _c in c_list:
            all_params.append(json.dumps([[], {'C': _c}]))
        self.all_params = all_params

    def _get_regression_layer_and_params(self, benchmark_builder):
        layer_and_params = []
        for layer in tqdm(self.layers, desc='Param layers'):
            _model_scores = RegressParamScores(
                    self.args.param_score_id or self.identifier, 
                    activations_model=self.activations_model)
            _score = _model_scores(
                    benchmark_builder=benchmark_builder, 
                    params=self.all_params,
                    layer=layer)
            layer_and_params.append(
                    [layer, 
                     self.all_params[
                         int(_score.sel(aggregation='center').argmax())]])
        self.layer_and_params = layer_and_params

    def hvm_it_mask_param_select_scores(self):
        self._get_activation_model()
        self._get_all_mask_params()
        self._get_layer_and_params(DicarloMajaj2015ITLowMidVarMask)
        score = self._run_layer_param_scores(DicarloMajaj2015ITMaskParams)
        print(score)

    def hvm_v4_mask_param_select_scores(self):
        self._get_activation_model()
        self._get_all_mask_params()
        self._get_layer_and_params(DicarloMajaj2015V4LowMidVarMask)
        score = self._run_layer_param_scores(DicarloMajaj2015V4MaskParams)
        print(score)

    def _get_layer_and_default_params(self):
        layer_and_params = []
        for layer in self.layers:
            layer_and_params.append(
                    [layer, json.dumps([[], DEFAULT_PARAMS])])
        self.layer_and_params = layer_and_params

    def hvm_it_mask_default_param_scores(self):
        self._get_activation_model()
        self._build_activations_model('-default')
        self._get_layer_and_default_params()
        score = self._run_layer_param_scores(DicarloMajaj2015ITMaskParams)
        print(score)

    def hvm_v4_mask_default_param_scores(self):
        self._get_activation_model()
        self._build_activations_model('-default')
        self._get_layer_and_default_params()
        score = self._run_layer_param_scores(DicarloMajaj2015V4MaskParams)
        print(score)

    def _set_v1_args(self):
        self.args.load_prep_type = 'v1_cadena'
        if self.args.ls_ld_range == 'default':
            self.args.ls_ld_range = 'v1_cadena_range'

    def cadena_v1_mask_default_param_scores(self):
        self._set_v1_args()
        self._get_activation_model()
        self._build_activations_model('-default')
        self._get_layer_and_default_params()
        score = self._run_layer_param_scores(ToliasCadena2017MaskParams)
        print(score)

    def cadena_v1_mask_param_select_scores(self):
        self._set_v1_args()
        self._get_activation_model()
        self._get_all_mask_params()
        self._get_layer_and_params(ToliasCadena2017MaskParams)
        score = self._run_layer_param_scores(ToliasCadena2017MaskParams)
        print(score)

    def cadena_v1_mask_param_select_cadena_scores(self):
        self._set_v1_args()
        self._get_activation_model()
        self._get_all_mask_params()
        self._get_layer_and_params(ToliasCadena2017MaskParams)
        score = self._run_layer_param_scores(ToliasCadena2017MaskParamsCadenaScores)
        print(score)

    def cadena_v1_param_select_all_cadena_scores(self):
        self._set_v1_args()
        self._get_activation_model()
        self._get_all_mask_params()
        self._get_layer_and_params(ToliasCadena2017MaskParamsCadenaScores)
        score = self._run_layer_param_scores(ToliasCadena2017MaskParamsCadenaScores)
        print(score)

    def cadena_v1_with_nans_mask_param_select_scores(self):
        self._set_v1_args()
        self._get_activation_model()
        self._get_all_mask_params()
        self._get_layer_and_params(ToliasCadena2017MaskParams)
        score = self._run_layer_param_scores(ToliasCadena2017WithNaNsMaskParams)
        print(score)

    def _get_layer_and_empty_params(self):
        layer_and_params = []
        for layer in self.layers:
            layer_and_params.append(
                    [layer, json.dumps([[], {}])])
        self.layer_and_params = layer_and_params

    def cadena_v1_correlation_scores(self):
        self._set_v1_args()
        self._get_activation_model()
        self._build_activations_model('-default')
        self._get_layer_and_empty_params()
        score = self._run_layer_param_scores(ToliasCadena2017Correlation)
        print(score)

    def cadena_v1_cadena_fit_cadena_scores(self):
        self._set_v1_args()
        self._get_activation_model()
        self._build_activations_model('-default')
        self._get_layer_and_empty_params()
        score = self._run_layer_param_scores(ToliasCadena2017CadenaFitCadenaScores)
        print(score)

    def cadena_v1_layer_param_pls_scores(self):
        self._get_activation_model()
        self._build_activations_model('-layer-param-v1')
        self._get_layer_and_empty_params()
        scores = self._run_layer_param_scores(
                lambda *args, **kwargs: bench.load('tolias.Cadena2017-pls'))
        print(scores)

    def hvm_it_layer_param_pls_scores(self):
        self._get_activation_model()
        self._build_activations_model('-layer-param-it')
        self._get_layer_and_empty_params()
        scores = self._run_layer_param_scores(
                lambda *args, **kwargs: bench.load('dicarlo.Majaj2015.IT-pls'))
        print(scores)

    def hvm_v4_layer_param_pls_scores(self):
        self._get_activation_model()
        self._build_activations_model('-layer-param-v4')
        self._get_layer_and_empty_params()
        scores = self._run_layer_param_scores(
                lambda *args, **kwargs: bench.load('dicarlo.Majaj2015.V4-pls'))
        print(scores)

    def objectome_i2n_layer_param_scores(self):
        self.dim_reduce = 'spatial_average'
        self._get_activation_model()
        self._build_activations_model('-layer-prob-run')
        scores = self._run_layer_probability_scores(
                bench.load('dicarlo.Rajalingham2018-i2n'))
        print(scores)

    def pca_objectome_i2n_layer_param_scores(self):
        self._get_activation_model()
        self._build_activations_model('-layer-prob-pca-run')
        LayerPCA.hook(self.activations_model, n_components=1000)
        scores = self._run_layer_probability_scores(
                bench.load('dicarlo.Rajalingham2018-i2n'))
        print(scores)

    def objectome_i2n_with_save_layer_param_scores(self):
        self.dim_reduce = 'spatial_average'
        self._get_activation_model()
        self._build_activations_model('-layer-prob-run-save')
        if self.args.pt_model and not self.args.pt_model_mid:
            # Use PCA hook TODO: finish the spatial average
            LayerPCA.hook(self.activations_model, n_components=1000)
        scores = self._run_layer_probability_scores_with_save(
                bhv_bench.DicarloRajalingham2018I2n_with_save)
        print(scores)

    def objectome_i2n_with_save_spearman_layer_param_scores(self):
        self.dim_reduce = 'spatial_average'
        self.i2n_save_suffix = '_spearman'
        self._get_activation_model()
        self._build_activations_model('-layer-prob-run-save')
        scores = self._run_layer_probability_scores_with_save(
                bhv_bench.DicarloRajalingham2018I2n_with_save_spearman)
        print(scores)

    def pca_trans_reg_layer_param_scores(self):
        self._get_activation_model()
        self._build_activations_model('-layer-reg-pca-run')
        LayerPCA.hook(self.activations_model, n_components=1000)
        #scores = self._run_layer_regression_scores(
        #        DicarloMajaj2015TransRegHighVar())
        self._get_layer_and_empty_params()
        scores = self._run_layer_regression_param_scores(
                DicarloMajaj2015TransRegHighVar)
        print(scores)

    def pca_trans_reg_layer_param_select_scores(self):
        self._get_activation_model()
        self._build_activations_model('-layer-reg-pca-run')
        LayerPCA.hook(self.activations_model, n_components=1000)
        self._get_all_regression_params()
        self._get_regression_layer_and_params(
                DicarloMajaj2015TransRegLowMidVar)
        score = self._run_layer_regression_param_scores(
                DicarloMajaj2015TransRegHighVar)
        print(score)

    def pca_rot_reg_layer_param_select_scores(self):
        # Not reported in the original paper
        self._get_activation_model()
        self._build_activations_model('-layer-reg-pca-run')
        LayerPCA.hook(self.activations_model, n_components=1000)
        self._get_all_regression_params()
        self._get_regression_layer_and_params(
                majaj2015_bench.DicarloMajaj2015RotRegLowMidVar)
        score = self._run_layer_regression_param_scores(
                majaj2015_bench.DicarloMajaj2015RotRegHighVar)
        print(score)

    def pca_rotsem_reg_layer_param_select_scores(self):
        self._get_activation_model()
        self._build_activations_model('-layer-reg-pca-run')
        LayerPCA.hook(self.activations_model, n_components=1000)
        self._get_all_regression_params()
        self._get_regression_layer_and_params(
                majaj2015_bench.DicarloMajaj2015RotSemRegLowMidVar)
        score = self._run_layer_regression_param_scores(
                majaj2015_bench.DicarloMajaj2015RotSemRegHighVar)
        print(score)

    def pca_area_reg_layer_param_select_scores(self):
        self._get_activation_model()
        self._build_activations_model('-layer-reg-pca-run')
        LayerPCA.hook(self.activations_model, n_components=1000)
        self._get_all_regression_params()
        self._get_regression_layer_and_params(
                majaj2015_bench.DicarloMajaj2015AreaRegLowMidVar)
        score = self._run_layer_regression_param_scores(
                majaj2015_bench.DicarloMajaj2015AreaRegHighVar)
        print(score)

    def pca_category_layer_param_select_scores(self):
        self._get_activation_model()
        self._build_activations_model('-layer-reg-pca-run')
        LayerPCA.hook(self.activations_model, n_components=1000)
        self._get_all_categorization_params()

        self._get_regression_layer_and_params(
                majaj2015_bench.DicarloMajaj2015CateLowMidVar)
        score = self._run_layer_regression_param_scores(
                majaj2015_bench.DicarloMajaj2015CateHighVar)
        print(score)

    def run(self):
        bench_func = getattr(self, self.args.bench_func)
        bench_func()


def main():
    parser = get_brainscore_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not (args.bs_models_id or args.pt_model or args.special_model):
        assert args.set_func, "Must specify set_func"
        args = load_set_func(args)

    score_vmmodels = ScoreVMModels(args)
    score_vmmodels.run()


if __name__ == '__main__':
    main()
