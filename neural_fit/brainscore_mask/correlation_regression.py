import numpy as np
import pdb
from scipy.stats import pearsonr
from tqdm import tqdm


class CorrelationRegression(object):
    def __init__(self, *args, **kwargs):
        self.best_indx = None

    def _normalize(self, Y):
        Y_raw_arr = np.asarray(Y)
        self._mean_Y = np.nanmean(Y_raw_arr, axis=0, keepdims=True)
        self._std_Y = np.nanstd(Y_raw_arr, axis=0, keepdims=True)
        Y = (Y_raw_arr - self._mean_Y) / self._std_Y
        return Y

    def fit(self, X, Y):
        normalize_X = self._normalize(X)
        normalize_Y = self._normalize(Y)
        all_corrs = np.matmul(normalize_X.transpose(), normalize_Y)
        self.best_indx = np.nanargmax(all_corrs, axis=0)

    def predict(self, X):
        assert not self.best_indx is None, "Must have fit before"
        return X[:, self.best_indx]
