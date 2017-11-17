# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for cross-validating hyperparameters"""
from __future__ import print_function, division, absolute_import
import numpy as np


def estimate_and_score(estimator, ftrain, ftest, strain, stest, regargs, **kwargs):
    """Estimate parameters and calculate score for a single fold"""
    estimator.select_data(ftrain, strain)
    l, a = regargs
    w = estimator.estimate(reg_lambda=l, reg_alpha=a, **kwargs)
    estimator.select_data(ftest, stest)
    # calculate score without the regularization penalty
    s = estimator.loglike(w)
    return w, -s


def estimate_crossval(estimator, n_splits, regargs, **kwargs):
    """Estimate parameters and calculate score using cross-validation"""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits)
    iter_frames = kf.split(estimator.X_stim)
    iter_bins = kf.split(estimator.X_spike)

    it = (estimate_and_score(estimator, ftrain, ftest, strain, stest, regargs, **kwargs)
          for (ftrain, ftest), (strain, stest) in zip(iter_frames, iter_bins))

    w, s = zip(*it)
    return np.mean(w, axis=0), np.sum(s)


def elasticnet(estimator, n_splits, alphas, l1_ratios=0.5, **kwargs):
    """Cross-validate using a regularization path

    estimator: (mle.estimator)
    n_splits: how many folds to split the data into

    alphas: an array of alpha values to test. Values should be log-spaced, from
       largest to smallest.

    l1_ratios: the ratio of l1 regularization to l2 regularization

    """
    # a lot of this is bastardized from scikit-learn, which doesn't know about
    # poisson error functions. We make the user specify what alphas to test
    if np.isscalar(l1_ratios):
        l1_ratios = [l1_ratios]
    w = None
    for l1r in l1_ratios:
        for alpha in alphas:
            regargs = (alpha * l1r, alpha * (1 - l1r))
            w, s = estimate_crossval(estimator, n_splits, regargs, w0=w, **kwargs)
            yield (regargs, s, w)


def find_best(iterator, disp=True):
    """Finds the values of regargs that maximize the score

    The iterator needs to yield (regargs), score, params

    Prints progress and returns the best value
    """
    results = []
    scores = []
    for regargs, s, w in iterator:
        if disp:
            print("{}: {}".format(regargs, s))
        scores.append(s)
        results.append((regargs, s, w))
    best = np.argmax(scores)
    return results[best]
