import brainscore
from brainio_base.assemblies import walk_coords, array_is_element
from brainscore.benchmarks._neural_common import NeuralBenchmark
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, pls_regression, \
    pearsonr_correlation
from brainscore.benchmarks.cadena2017 import AssemblyLoader
from brainscore_mask.faster_mask_regression import FasterMaskRegression
from brainscore.metrics.regression import XarrayRegression, XarrayCorrelation
import pdb
import numpy as np
from brainscore.metrics import Score
from cadena_regression import CadenaRegression
from correlation_regression import CorrelationRegression


def mask_regression_with_params(*args, **kwargs):
    regression = FasterMaskRegression(*args, **kwargs)
    regression = XarrayRegression(regression)
    return regression


def cadena_regression(*args, **kwargs):
    regression = CadenaRegression(*args, **kwargs)
    regression = XarrayRegression(regression)
    return regression


def corr_regression(*args, **kwargs):
    regression = CorrelationRegression(*args, **kwargs)
    regression = XarrayRegression(regression)
    return regression


def ToliasCadena2017Correlation(*regression_args, **regression_kwargs):
    loader = AssemblyLoader()
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.name = assembly.stimulus_set_name

    similarity_metric = CrossRegressedCorrelation(
        regression=corr_regression(
            *regression_args, **regression_kwargs), 
        correlation=pearsonr_correlation(),
        crossvalidation_kwargs={'splits': 4, 'stratification_coord': None})
    identifier = f'tolias.Cadena2017-correlation'
    ceiler = InternalConsistency(split_coord='repetition_id')

    def ceiling():
        assembly_nonan, stimuli = loader.dropna(
                assembly_repetition, 
                assembly_repetition.attrs['stimulus_set'])
        return ceiler(assembly_nonan)
    return NeuralBenchmark(identifier=identifier, version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=ceiling)


def ToliasCadena2017MaskParams(*regression_args, **regression_kwargs):
    loader = AssemblyLoader()
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.name = assembly.stimulus_set_name

    similarity_metric = CrossRegressedCorrelation(
        regression=mask_regression_with_params(
            *regression_args, **regression_kwargs), 
        correlation=pearsonr_correlation(),
        crossvalidation_kwargs={'splits': 4, 'stratification_coord': None})
    identifier = f'tolias.Cadena2017-param-mask'
    ceiler = InternalConsistency(split_coord='repetition_id')

    def ceiling():
        assembly_nonan, stimuli = loader.dropna(
                assembly_repetition, 
                assembly_repetition.attrs['stimulus_set'])
        return ceiler(assembly_nonan)
    return NeuralBenchmark(identifier=identifier, version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=ceiling)


class AssemblyLoaderWithNaN:
    name = 'tolias.Cadena2017.all'

    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name='tolias.Cadena2017')
        assembly = assembly.rename({'neuroid': 'neuroid_id'}).stack(neuroid=['neuroid_id'])
        assembly.load()
        assembly['region'] = 'neuroid', ['V1'] * len(assembly['neuroid'])
        assembly = assembly.squeeze("time_bin")
        assembly = assembly.transpose('presentation', 'neuroid')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly

    def average_repetition(self, assembly):
        attrs = assembly.attrs  # workaround to keeping attrs
        presentation_coords = [coord for coord, dims, values in walk_coords(assembly)
                               if array_is_element(dims, 'presentation')]
        presentation_coords = set(presentation_coords) - {'repetition_id', 'id'}
        assembly = assembly.multi_groupby(presentation_coords).mean(dim='presentation', skipna=True)
        assembly.attrs = attrs
        return assembly


def cadena_rep_score(xarray_kwargs=None):
    xarray_kwargs = xarray_kwargs or {}
    def rep_score_func(target, prediction):
        target = np.asarray(target)
        prediction = np.asarray(prediction)
        tot_var = np.nanvar(target)
        left_var = np.nanmean((prediction - target) ** 2)
        exp_var = (tot_var - left_var) / tot_var
        return exp_var, 0
    return XarrayCorrelation(rep_score_func, **xarray_kwargs)


def ToliasCadena2017MaskParamsCadenaScores(*regression_args, **regression_kwargs):
    loader = AssemblyLoaderWithNaN()
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.name = assembly.stimulus_set_name

    similarity_metric = CrossRegressedCorrelation(
        regression=mask_regression_with_params(
            *regression_args, **regression_kwargs), 
        correlation=cadena_rep_score(),
        crossvalidation_kwargs={
            'train_size': .8, 'splits': 4, 'stratification_coord': None})
    identifier = f'tolias.Cadena2017-param-mask-cadena-scores'
    ceiling_func = lambda: Score(
            [1, np.nan], 
            coords={'aggregation': ['center', 'error']}, 
            dims=['aggregation'])

    return NeuralBenchmark(identifier=identifier, version=1,
                           assembly=assembly, 
                           similarity_metric=similarity_metric,
                           ceiling_func=ceiling_func)


def ToliasCadena2017CadenaFitCadenaScores(*regression_args, **regression_kwargs):
    loader = AssemblyLoaderWithNaN()
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.name = assembly.stimulus_set_name

    similarity_metric = CrossRegressedCorrelation(
        regression=cadena_regression(
            *regression_args, **regression_kwargs), 
        correlation=cadena_rep_score(),
        crossvalidation_kwargs={
            'train_size': .8, 'splits': 2, 'stratification_coord': None})
    identifier = f'tolias.Cadena2017-cadena-fit-cadena-scores'
    ceiling_func = lambda: Score(
            [1, np.nan], 
            coords={'aggregation': ['center', 'error']}, 
            dims=['aggregation'])

    return NeuralBenchmark(identifier=identifier, version=1,
                           assembly=assembly, 
                           similarity_metric=similarity_metric,
                           ceiling_func=ceiling_func)


def nan_pearsonr_correlation(xarray_kwargs=None):
    xarray_kwargs = xarray_kwargs or {}
    def nan_pearsonr_func(target, prediction):
        from scipy.stats import pearsonr
        target = np.asarray(target)
        prediction = np.asarray(prediction)
        nan_mask = np.isnan(target)
        real_mask = np.logical_not(nan_mask)
        target = target[real_mask]
        prediction = prediction[real_mask]
        return pearsonr(target, prediction)
    return XarrayCorrelation(nan_pearsonr_func, **xarray_kwargs)


def ToliasCadena2017WithNaNsMaskParams(*regression_args, **regression_kwargs):
    loader = AssemblyLoader()
    assembly_repetition = loader(average_repetition=False)
    loader_with_nan = AssemblyLoaderWithNaN()
    assembly = loader_with_nan(average_repetition=True)
    assembly.stimulus_set.name = assembly.stimulus_set_name

    similarity_metric = CrossRegressedCorrelation(
        regression=mask_regression_with_params(
            *regression_args, **regression_kwargs), 
        correlation=nan_pearsonr_correlation(),
        crossvalidation_kwargs={
            'train_size': .8, 'splits': 4, 'stratification_coord': None})
    identifier = f'tolias.Cadena2017-with-nans-param-mask'
    ceiler = InternalConsistency(split_coord='repetition_id')

    def ceiling():
        assembly_nonan, stimuli = loader.dropna(
                assembly_repetition, 
                assembly_repetition.attrs['stimulus_set'])
        return ceiler(assembly_nonan)
    return NeuralBenchmark(identifier=identifier, version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=ceiling)
