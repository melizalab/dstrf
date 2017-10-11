# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for spectrotemporal receptive fields"""
from __future__ import print_function, division, absolute_import

import numpy as np
import scipy as sp


def lagged_matrix(spec, ntau):
    """Convert a (nfreq, nt) spectrogram into a (nt, nfreq*ntau) design matrix"""
    from scipy.linalg import hankel
    if spec.ndim == 1:
        spec = np.expand_dims(spec, 0)
    nf, nt = spec.shape
    X = np.zeros((nt, nf * ntau), dtype=spec.dtype)
    padding = np.zeros(ntau - 1, dtype=spec.dtype)
    for i in range(nf):
        h = hankel(np.concatenate([padding, spec[i, :(-ntau + 1)]]), spec[i, -ntau:])
        X[:, (i * ntau):((i + 1) * ntau)] = h
    return X


def as_vector(strf):
    """Convert a (nfreq, ntau) kernel to a vector that can be used for matrix convolution

    To be compatible with the output of lagged_matrix, the kernel must be
    time-inverted (i.e., large tau indices are short lags)

    """
    return np.ravel(strf, order='C')


def as_matrix(strf, ntau):
    """Convert an (nfreq * ntau,) kernel back into (nfreq, ntau) form"""
    nfreq = strf.size // ntau
    return strf.reshape(nfreq, ntau)


def correlate(stim_design, spikes):
    """Calculate correlation between stim (as design matrix) and spikes (i.e., spike-triggered average)"""
    nframes, nfeat = stim_design.shape
    nbins = spikes.size
    upsample = nbins // nframes
    # coarse binning of stimulus
    psth = np.sum(spikes.reshape(nframes, upsample), axis=1)
    return np.dot(stim_design.T, psth) / np.sum(psth)


def subspace(rf1, rf2):
    """Calculate the angle between two RFs as a measure of error"""
    cos_theta = np.sum(rf1 * rf2) / np.sqrt(np.sum(rf1 ** 2) * np.sum(rf2 ** 2))
    return np.acos(cos_theta)
