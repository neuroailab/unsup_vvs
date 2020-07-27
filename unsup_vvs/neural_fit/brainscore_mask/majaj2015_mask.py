import brainscore
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.benchmarks.majaj2015 import _DicarloMajaj2015Region, load_assembly
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, pearsonr_correlation
from brainscore.utils import LazyLoad
from brainscore.metrics.regression import MaskRegression, XarrayRegression, XarrayCorrelation
from brainscore_mask.faster_mask_regression import FasterMaskRegression
from brainscore.benchmarks import BenchmarkBase
import pdb
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
import numpy as np
from brainio_base.assemblies import NeuroidAssembly, array_is_element, walk_coords


class RegressionBenchmark(BenchmarkBase):
    def __init__(
            self, identifier, assembly, similarity_metric, 
            **kwargs):
        super(RegressionBenchmark, self).__init__(identifier=identifier, **kwargs)
        self._assembly = assembly
        self._similarity_metric = similarity_metric

    def __call__(self, candidate):
        source_assembly = candidate.look_at(self._assembly.stimulus_set)
        source_assembly = source_assembly.transpose('presentation', 'neuroid')
        raw_score = self._similarity_metric(source_assembly, self._assembly)
        return raw_score


def build_assembly(assembly, coord_list=['ty', 'tz']):
    values = np.stack(
            [getattr(assembly, coord).values for coord in coord_list], 
            axis=1)
    coords = {
            'neuroid_id': ('neuroid', list(range(len(coord_list)))),
            'neuroid_meaning': ('neuroid', coord_list)}
    for coord, dims, value in walk_coords(assembly):
        if len(dims) == 0:
            continue
        if dims[0] == 'presentation':
            coords[coord] = ('presentation', value)
    new_assembly = NeuroidAssembly(
            values,
            coords=coords,
            dims=['presentation', 'neuroid'])
    new_assembly.attrs['stimulus_set'] = assembly.stimulus_set
    return new_assembly


def _DicarloMajaj2015RegLowMidVar(
        identifier_metric_suffix, similarity_metric,
        coord_list):
    assembly = load_var0var3_assembly(average_repetitions=True, region='V4')
    assembly = build_assembly(assembly, coord_list)
    return RegressionBenchmark(
            identifier=f'dicarlo.Majaj2015.lowmidvar.reg.{identifier_metric_suffix}', 
            version=3,
            assembly=assembly, similarity_metric=similarity_metric,
            ceiling_func=None,
            paper_link='http://www.jneurosci.org/content/35/39/13402.short')


def _DicarloMajaj2015RegHighVar(
        identifier_metric_suffix, similarity_metric, coord_list):
    assembly = load_assembly(average_repetitions=True, region='V4')
    assembly = build_assembly(assembly, coord_list)
    return RegressionBenchmark(
            identifier=f'dicarlo.Majaj2015.highvar.reg.{identifier_metric_suffix}', 
            version=3,
            assembly=assembly, similarity_metric=similarity_metric,
            ceiling_func=None,
            paper_link='http://www.jneurosci.org/content/35/39/13402.short')


def DicarloMajaj2015TransRegLowMidVar(*regression_args, **regression_kwargs):
    return _DicarloMajaj2015RegLowMidVar(
            identifier_metric_suffix='var03_trans',
            coord_list=['ty', 'tz'],
            similarity_metric=ScaledCrossRegressedCorrelation(
                regression=XarrayRegression(
                    MultiOutputRegressor(LinearSVR(
                        *regression_args, **regression_kwargs))), 
                correlation=pearsonr_correlation(),
                crossvalidation_kwargs=dict(
                    splits=4, stratification_coord='object_name')))


def DicarloMajaj2015TransRegHighVar(*regression_args, **regression_kwargs):
    return _DicarloMajaj2015RegHighVar(
            identifier_metric_suffix='var6_trans',
            coord_list=['ty', 'tz'],
            similarity_metric=ScaledCrossRegressedCorrelation(
                regression=XarrayRegression(
                    MultiOutputRegressor(LinearSVR(
                        *regression_args, **regression_kwargs))), 
                correlation=pearsonr_correlation(),
                crossvalidation_kwargs=dict(
                    splits=4, stratification_coord='object_name')))


def DicarloMajaj2015RotRegLowMidVar(*regression_args, **regression_kwargs):
    return _DicarloMajaj2015RegLowMidVar(
            identifier_metric_suffix='var03_rot',
            coord_list=['rxy', 'ryz', 'rxz'],
            similarity_metric=ScaledCrossRegressedCorrelation(
                regression=XarrayRegression(
                    MultiOutputRegressor(LinearSVR(
                        *regression_args, **regression_kwargs))), 
                correlation=pearsonr_correlation(),
                crossvalidation_kwargs=dict(
                    splits=4, stratification_coord='object_name')))


def DicarloMajaj2015RotRegHighVar(*regression_args, **regression_kwargs):
    return _DicarloMajaj2015RegHighVar(
            identifier_metric_suffix='var6_rot',
            coord_list=['rxy', 'ryz', 'rxz'],
            similarity_metric=ScaledCrossRegressedCorrelation(
                regression=XarrayRegression(
                    MultiOutputRegressor(LinearSVR(
                        *regression_args, **regression_kwargs))), 
                correlation=pearsonr_correlation(),
                crossvalidation_kwargs=dict(
                    splits=4, stratification_coord='object_name')))


def DicarloMajaj2015RotSemRegLowMidVar(*regression_args, **regression_kwargs):
    return _DicarloMajaj2015RegLowMidVar(
            identifier_metric_suffix='var03_rotsem',
            coord_list=['rxy_semantic', 'ryz_semantic', 'rxz_semantic'],
            similarity_metric=ScaledCrossRegressedCorrelation(
                regression=XarrayRegression(
                    MultiOutputRegressor(LinearSVR(
                        *regression_args, **regression_kwargs))), 
                correlation=pearsonr_correlation(),
                crossvalidation_kwargs=dict(
                    splits=4, stratification_coord='object_name')))


def DicarloMajaj2015RotSemRegHighVar(*regression_args, **regression_kwargs):
    return _DicarloMajaj2015RegHighVar(
            identifier_metric_suffix='var6_rotsem',
            coord_list=['rxy_semantic', 'ryz_semantic', 'rxz_semantic'],
            similarity_metric=ScaledCrossRegressedCorrelation(
                regression=XarrayRegression(
                    MultiOutputRegressor(LinearSVR(
                        *regression_args, **regression_kwargs))), 
                correlation=pearsonr_correlation(),
                crossvalidation_kwargs=dict(
                    splits=4, stratification_coord='object_name')))


def DicarloMajaj2015AreaRegLowMidVar(*regression_args, **regression_kwargs):
    return _DicarloMajaj2015RegLowMidVar(
            identifier_metric_suffix='var03_area',
            coord_list=['s'],
            similarity_metric=ScaledCrossRegressedCorrelation(
                regression=XarrayRegression(
                    MultiOutputRegressor(LinearSVR(
                        *regression_args, **regression_kwargs))), 
                correlation=pearsonr_correlation(),
                crossvalidation_kwargs=dict(
                    splits=4, stratification_coord='object_name')))


def DicarloMajaj2015AreaRegHighVar(*regression_args, **regression_kwargs):
    return _DicarloMajaj2015RegHighVar(
            identifier_metric_suffix='var6_area',
            coord_list=['s'],
            similarity_metric=ScaledCrossRegressedCorrelation(
                regression=XarrayRegression(
                    MultiOutputRegressor(LinearSVR(
                        *regression_args, **regression_kwargs))), 
                correlation=pearsonr_correlation(),
                crossvalidation_kwargs=dict(
                    splits=4, stratification_coord='object_name')))


def build_cate_assembly(assembly):
    category_names = assembly.category_name.values
    unique_cate_names = np.unique(category_names)
    # Tricky solution for some weird requirements later
    new_category_names = [
            [curr_name, curr_name] \
            for curr_name in category_names]
    coords = {
            'neuroid_id': ('neuroid', [0, 1]),
            'neuroid_meaning': ('neuroid', ['category', 'category'])}
    for coord, dims, value in walk_coords(assembly):
        if len(dims) == 0:
            continue
        if dims[0] == 'presentation':
            coords[coord] = ('presentation', value)
    new_assembly = NeuroidAssembly(
            new_category_names,
            coords=coords,
            dims=['presentation', 'neuroid'])
    new_assembly.attrs['stimulus_set'] = assembly.stimulus_set
    return new_assembly


def tricky_accuracy(xarray_kwargs=None):
    xarray_kwargs = xarray_kwargs or {}
    return XarrayCorrelation(
            lambda x, y: (np.sum(x.values == y.values) / len(x), None), **xarray_kwargs)


def DicarloMajaj2015CateLowMidVar(*regression_args, **regression_kwargs):
    assembly = load_var0var3_assembly(average_repetitions=True, region='V4')
    assembly = build_cate_assembly(assembly)
    return RegressionBenchmark(
            identifier='dicarlo.Majaj2015.lowmidvar.cate', 
            assembly=assembly,
            similarity_metric=CrossRegressedCorrelation(
                regression=XarrayRegression(
                    MultiOutputClassifier(LinearSVC(
                        *regression_args, **regression_kwargs))), 
                correlation=tricky_accuracy(),
                crossvalidation_kwargs=dict(
                    splits=4, stratification_coord='object_name')),
            version=3, ceiling_func=None,
            paper_link='http://www.jneurosci.org/content/35/39/13402.short',
            )


def DicarloMajaj2015CateHighVar(*regression_args, **regression_kwargs):
    assembly = load_assembly(average_repetitions=True, region='V4')
    assembly = build_cate_assembly(assembly)
    return RegressionBenchmark(
            identifier='dicarlo.Majaj2015.highvar.cate', 
            assembly=assembly,
            similarity_metric=CrossRegressedCorrelation(
                regression=XarrayRegression(
                    MultiOutputClassifier(LinearSVC(
                        *regression_args, **regression_kwargs))), 
                correlation=tricky_accuracy(),
                crossvalidation_kwargs=dict(
                    splits=4, stratification_coord='object_name')),
            version=3, ceiling_func=None,
            paper_link='http://www.jneurosci.org/content/35/39/13402.short',
            )


def _DicarloMajaj2015RegionLowMidVar(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = LazyLoad(lambda region=region: load_var0var3_assembly(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_var0var3_assembly(average_repetitions=True, region=region))
    return NeuralBenchmark(identifier=f'dicarlo.Majaj2015.lowmidvar.{region}-{identifier_metric_suffix}', version=3,
                           assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region, paper_link='http://www.jneurosci.org/content/35/39/13402.short')


def mask_regression_with_params(*args, **kwargs):
    regression = FasterMaskRegression(*args, **kwargs)
    regression = XarrayRegression(regression)
    return regression


def DicarloMajaj2015ITLowMidVarMask(*regression_args, **regression_kwargs):
    return _DicarloMajaj2015RegionLowMidVar('IT', identifier_metric_suffix='var03_param_mask',
                                   similarity_metric=ScaledCrossRegressedCorrelation(
                                       regression=mask_regression_with_params(
                                           *regression_args, **regression_kwargs), 
                                       correlation=pearsonr_correlation(),
                                       crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')),
                                   ceiler=InternalConsistency())


def DicarloMajaj2015ITMaskParams(*regression_args, **regression_kwargs):
    return _DicarloMajaj2015Region('IT', identifier_metric_suffix='param_mask',
                                   similarity_metric=ScaledCrossRegressedCorrelation(
                                       regression=mask_regression_with_params(
                                           *regression_args, **regression_kwargs), 
                                       correlation=pearsonr_correlation(),
                                       crossvalidation_kwargs=dict(splits=4, stratification_coord='object_name')),
                                   ceiler=InternalConsistency())


def DicarloMajaj2015V4LowMidVarMask(*regression_args, **regression_kwargs):
    return _DicarloMajaj2015RegionLowMidVar('V4', identifier_metric_suffix='var03_param_mask',
                                   similarity_metric=ScaledCrossRegressedCorrelation(
                                       regression=mask_regression_with_params(
                                           *regression_args, **regression_kwargs), 
                                       correlation=pearsonr_correlation(),
                                       crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')),
                                   ceiler=InternalConsistency())


def DicarloMajaj2015V4MaskParams(*regression_args, **regression_kwargs):
    return _DicarloMajaj2015Region('V4', identifier_metric_suffix='param_mask',
                                   similarity_metric=ScaledCrossRegressedCorrelation(
                                       regression=mask_regression_with_params(
                                           *regression_args, **regression_kwargs), 
                                       correlation=pearsonr_correlation(),
                                       crossvalidation_kwargs=dict(splits=4, stratification_coord='object_name')),
                                   ceiler=InternalConsistency())


def load_var0var3_assembly(average_repetitions, region):
    assembly = brainscore.get_assembly(name=f'dicarlo.Majaj2015.public')
    assembly = assembly.sel(region=region)
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly = assembly.squeeze("time_bin")
    assembly.load()
    assembly = assembly.transpose('presentation', 'neuroid')
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly


if __name__ == '__main__':
    #majaj2015_it_all = load_var3_assembly(True, 'IT')
    majaj2015_it_all = load_var3_assembly(False, 'IT')
    pdb.set_trace()
