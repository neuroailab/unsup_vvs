from __future__ import division
import numpy as np
from sklearn import svm


def smooth(vals, conv_len):
    conv_list = np.ones([conv_len])/conv_len
    return np.convolve(vals, conv_list, mode='valid')


def smooth_vecs(vecs, conv_len):
    return np.apply_along_axis(
        (lambda v: smooth(v, conv_len)),
        axis=0, arr=vecs)


def pca_analysis(vecs, topk=10,
                 shift_by_mean=False):
    assert len(vecs.shape) == 2
    # usually, vecs[0] is an embedding vector

    # gauss = np.random.randn(*vecs.shape)
    # u, s, v = np.linalg.svd(gauss, full_matrices=False)
    # control_s = s * s
    # control_s /= sum(control_s)

    if shift_by_mean:
        avg = np.mean(vecs, axis=0)
        assert len(avg.shape) == 1
        vecs = vecs - avg

    u, s, v = np.linalg.svd(vecs.T, full_matrices=False)
    s = s * s
    tot_var = sum(s)
    s /= tot_var
    return s[:topk], tot_var, u.T[:topk]


def fit_least_squares_transformation(xs, ys):
    # Returns a transform that when *left multiplied* against column
    # vectors xs yields an estimate of the ys
    assert xs.shape == ys.shape
    n, dim = xs.shape
    xform, resid, _, _ = np.linalg.lstsq(xs, ys)
    assert xform.shape == (dim, dim)
    print(np.sqrt(np.sum(resid)))
    return xform.T


def linsep(vecs, against_vec, threshold):
    v = against_vec / np.linalg.norm(against_vec)
    dps = np.dot(vecs, v)
    pos_ct = np.sum(dps > threshold)
    return pos_ct / len(vecs)


class LinearSeparator(object):
    def __init__(self, cloud1, cloud2):
        self.k1, self.dim = cloud1.shape
        self.k2, self.dim = cloud2.shape
        self.cloud1 = cloud1
        self.cloud2 = cloud2

    def separate_svm(self):
        sep = svm.LinearSVC()
        X = np.concatenate((self.cloud1, self.cloud2))
        y = [1] * self.k1 + [2] * self.k2
        sep.fit(X, y)
        return sep.score(X, y), sep.coef_[0], sep.intercept_[0]

    def separate_along_vec(self, along_vec):
        dps1 = np.dot(self.cloud1, along_vec)
        dps2 = np.dot(self.cloud2, along_vec)
        dp_pairs = [(x, 1) for x in dps1] + [(x, 2) for x in dps2]
        dp_pairs = sorted(dp_pairs)
        N = len(dp_pairs)

        best_score = 0.0
        best_idx = -1

        ct1 = ct2 = 0
        for i in range(N):
            _, which = dp_pairs[i]
            if which == 1:
                ct1 += 1
            else:
                ct2 += 1
            if i < N/10 or i > 9*N/10:
                continue

            rest1 = len(dps1) - ct1
            rest2 = len(dps2) - ct2
            score = min(ct1 / (ct1 + ct2),
                        rest2 / (rest1 + rest2))
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx, best_score
