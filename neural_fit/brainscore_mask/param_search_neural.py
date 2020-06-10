import logging
from collections import Iterable
from typing import Optional, Union

from tqdm import tqdm
import pdb

from brainscore.metrics import Score
from brainscore.model_interface import BrainModel
from brainscore.utils import fullname
from model_tools.activations.pca import LayerPCA
from model_tools.brain_transformation import TemporalIgnore
from result_caching import store_xarray, store
import json
from model_tools.brain_transformation import LayerMappedModel


class LayerModel(BrainModel):
    def __init__(self, identifier, activations_model, layers):
        self.identifier = identifier
        self.activations_model = activations_model
        self.layers = layers

    def look_at(self, stimuli):
        activations = self.activations_model(stimuli, layers=self.layers)
        return activations


class ParamScores:
    def __init__(self, model_identifier, activations_model):
        self.model_identifier = model_identifier
        self._activations_model = activations_model
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, benchmark_builder, params, layer, prerun=False):
        # Prerun is not supported as different benchmarks can have different inputs (although unlikely)
        assert not prerun, "Prerun not supported for ParamScores"
        benchmark_identifier = benchmark_builder().identifier
        return self._call(model_identifier=self.model_identifier,
                          model=self._activations_model, layer=layer, 
                          benchmark_identifier=benchmark_identifier,
                          benchmark_builder=benchmark_builder, params=params)

    def build_layer_model(
            self, identifier, model, benchmark, layer):
        layer_model = LayerMappedModel(
                identifier=identifier,
                activations_model=model, 
                region_layer_map={benchmark.region: layer})
        layer_model = TemporalIgnore(layer_model)
        return layer_model

    @store_xarray(
            identifier_ignore=['model', 'benchmark_builder', 'params'], 
            combine_fields={'params': 'param'})
    def _call(self, model_identifier, layer, benchmark_identifier, # storage fields
              model, benchmark_builder, params):
        param_scores = []
        for param_str in tqdm(params, desc="params"):
            bench_args, bench_kwargs = json.loads(param_str)
            benchmark = benchmark_builder(*bench_args, **bench_kwargs)
            layer_model = self.build_layer_model(
                    identifier=f"{model_identifier}-{layer}",
                    model=model, benchmark=benchmark, layer=layer,
                    )
            score = benchmark(layer_model)
            score = score.expand_dims('param')
            score['param'] = [param_str]
            param_scores.append(score)
        param_scores = Score.merge(*param_scores)
        param_scores = param_scores.sel(param=params)  # preserve layer ordering
        return param_scores


class RegressParamScores(ParamScores):
    def build_layer_model(
            self, identifier, model, benchmark, layer):
        layer_model = LayerModel(
                identifier,
                model,
                [layer])
        return layer_model


class LayerParamScores:
    def __init__(self, model_identifier, activations_model):
        self.model_identifier = model_identifier
        self._activations_model = activations_model
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, benchmark_builder, layer_and_params, prerun=False):
        # Params should be a tuple of (args, kwargs)
        # Prerun is not supported as different benchmarks can have different inputs (although unlikely)
        assert not prerun, "Prerun not supported for LayerParamScores"

        benchmark_identifier = benchmark_builder().identifier
        return self._call(model_identifier=self.model_identifier,
                          model=self._activations_model, 
                          benchmark_builder=benchmark_builder,
                          benchmark_identifier=benchmark_identifier,
                          layer_and_params=layer_and_params)

    def build_layer_model(
            self, identifier, model, benchmark, layer):
        layer_model = LayerMappedModel(
                identifier=identifier,
                activations_model=model, 
                region_layer_map={benchmark.region: layer})
        layer_model = TemporalIgnore(layer_model)
        return layer_model

    @store_xarray(
            identifier_ignore=['model', 'benchmark_builder', 'layer_and_params'], 
            combine_fields={'layers': 'layer', 'params': 'param'})
    def _call(self, model_identifier, benchmark_identifier,  # storage fields
              model, benchmark_builder, layer_and_params):
        all_scores = []
        all_layer_param_str = []
        for layer, param_str in tqdm(layer_and_params, desc="layers"):
            bench_args, bench_kwargs = json.loads(param_str)
            benchmark = benchmark_builder(*bench_args, **bench_kwargs)
            layer_model = self.build_layer_model(
                    identifier=f"{model_identifier}-{layer}-layer-param",
                    model=model, benchmark=benchmark, layer=layer,
                    )
            score = benchmark(layer_model)
            score = score.expand_dims('layer_param')
            layer_param_str = '%s-%s' % (layer, param_str)
            score['layer_param'] = [layer_param_str]
            all_scores.append(score)
            all_layer_param_str.append(layer_param_str)
        all_scores = Score.merge(*all_scores)
        all_scores = all_scores.sel(layer_param=all_layer_param_str)
        return all_scores


class LayerRegressParamScores(LayerParamScores):
    def build_layer_model(
            self, identifier, model, benchmark, layer):
        layer_model = LayerModel(
                identifier,
                model,
                [layer])
        return layer_model


class LayerActivations:
    def __init__(self, model_identifier, activations_model):
        self.model_identifier = model_identifier
        self._activations_model = activations_model
        self._logger = logging.getLogger(fullname(self))

    def __call__(
            self, benchmark, layers, benchmark_identifier=None, prerun=False):
        return self._call(
                model_identifier=self.model_identifier,
                benchmark_identifier=benchmark_identifier or benchmark.identifier,
                model=self._activations_model, 
                benchmark=benchmark, layers=layers, prerun=prerun)

    @store_xarray(
            identifier_ignore=['model', 'benchmark', 'layers', 'prerun'], 
            combine_fields={'layers': 'layer'})
    def _call(self, model_identifier, benchmark_identifier,  # storage fields
              model, benchmark, layers, prerun=False):
        if prerun:
            # pre-run activations together to avoid running every layer separately
            model(layers=layers, stimuli=benchmark._assembly.stimulus_set)

        for layer in tqdm(layers, desc="layers"):
            layer_model = LayerMappedModel(
                    identifier=f"{model_identifier}-{layer}",
                    activations_model=model, 
                    region_layer_map={benchmark.region: layer})
            layer_model = TemporalIgnore(layer_model)
            layer_model.start_recording(
                    benchmark.region, 
                    time_bins=benchmark.timebins)
            layer_model.look_at(benchmark._assembly.stimulus_set)
