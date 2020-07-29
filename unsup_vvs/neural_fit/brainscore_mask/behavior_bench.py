import numpy as np

import brainscore
from brainscore.metrics.image_level_behavior import _I
from brainscore.benchmarks.rajalingham2018 import DicarloRajalingham2018I2n
from brainscore.metrics import Metric, Score
from brainscore.metrics.transformations import apply_aggregate
from scipy.stats import pearsonr, spearmanr
import pickle


class DicarloRajalingham2018I2n_with_save(DicarloRajalingham2018I2n):
    def __init__(self, pkl_path):
        super().__init__()
        self._metric = I2n_with_save(pkl_path=pkl_path)


class DicarloRajalingham2018I2n_with_save_spearman(DicarloRajalingham2018I2n):
    def __init__(self, pkl_path):
        super().__init__()
        self._metric = I2n_with_save_spearman(pkl_path=pkl_path)


def I2n_with_save(*args, **kwargs):
    return _I_with_save(
            *args, collapse_distractors=False, normalize=True, **kwargs)


def I2n_with_save_spearman(*args, **kwargs):
    return _I_with_save(
            *args, 
            collapse_distractors=False, normalize=True, use_pearsonr=False,
            **kwargs)


class _I_with_save(_I):
    def __init__(self, pkl_path, use_pearsonr=True, *args, **kwargs):
        self.pkl_path = pkl_path
        self.all_source_matrix = []
        self.all_target_matrix = []
        self.all_correlation = []
        self.use_pearsonr = use_pearsonr
        super().__init__(*args, **kwargs)

    @classmethod
    def correlate(
            cls, source_response_matrix, target_response_matrix, 
            skipna=False, use_pearsonr=True):
        # align
        source_response_matrix = source_response_matrix.sortby('image_id').sortby('choice')
        target_response_matrix = target_response_matrix.sortby('image_id').sortby('choice')
        assert all(source_response_matrix['image_id'].values == target_response_matrix['image_id'].values)
        assert all(source_response_matrix['choice'].values == target_response_matrix['choice'].values)
        # flatten and mask out NaNs
        source, target = source_response_matrix.values.flatten(), target_response_matrix.values.flatten()
        non_nan = ~np.isnan(target)
        non_nan = np.logical_and(non_nan, (~np.isnan(source) if skipna else 1))
        source, target = source[non_nan], target[non_nan]
        assert not any(np.isnan(source))
        if use_pearsonr:
            correlation, p = pearsonr(source, target)
        else:
            correlation, p = spearmanr(source, target)
        return correlation

    def _call_single(self, source_probabilities, target, random_state):
        self.add_source_meta(source_probabilities, target)
        source_response_matrix = self.target_distractor_scores(source_probabilities)
        source_response_matrix = self.dprimes(source_response_matrix)
        if self._collapse_distractors:
            source_response_matrix = self.collapse_distractors(source_response_matrix)

        target_half = self.generate_halves(target, random_state=random_state)[0]
        target_response_matrix = self.build_response_matrix_from_responses(target_half)
        target_response_matrix = self.dprimes(target_response_matrix)
        if self._collapse_distractors:
            target_response_matrix = self.collapse_distractors(target_response_matrix)
            raise NotImplementedError("correlation for I1 not implemented")

        correlation = self.correlate(
                source_response_matrix, target_response_matrix,
                use_pearsonr=self.use_pearsonr)
        self._put_matrix(
                source_response_matrix, target_response_matrix, correlation)
        return correlation

    def _put_matrix(
            self, source_response_matrix, target_response_matrix, correlation):
        self.all_source_matrix.append(
                source_response_matrix.sortby('image_id').sortby('choice'))
        self.all_target_matrix.append(
                target_response_matrix.sortby('image_id').sortby('choice'))
        self.all_correlation.append(correlation)

    def _repeat(self, func):
        random_state = self._initialize_random_state()
        repetitions = list(range(self._repetitions))
        scores = [func(random_state=random_state) for repetition in repetitions]
        score = Score(scores, coords={'split': repetitions}, dims=['split'])
        self._save_matrix()
        return apply_aggregate(self.aggregate, score)

    def _save_matrix(self):
        with open(self.pkl_path, 'wb') as fout:
            result_dict = {
                    'source': self.all_source_matrix,
                    'target': self.all_target_matrix,
                    'correlation': self.all_correlation,
                    }
            pickle.dump(result_dict, fout)
