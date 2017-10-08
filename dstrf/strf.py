# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for spectrotemporal receptive fields"""
from __future__ import print_function, division, absolute_import

import numpy as np
import scipy as sp


def lagged_matrix(spec, ntau):
    """Convert a (nfreq, nt) spectrogram into a (nt, nfreq*ntau) design matrix"""
    from scipy.linalg import hankel
    nf, nt = spec.shape
    X = np.zeros((nt, nf * ntau), dtype=spec.dtype)
    padding = np.zeros(ntau - 1, dtype=spec.dtype)
    for i in range(nf):
        h = hankel(np.concatenate([padding, spec[i, :(-ntau + 1)]]), spec[i, -ntau:])
        X[:, (i * ntau):((i + 1) * ntau)] = h
    return X


def as_vector(strf):
    """Convert a (nfreq, ntau) kernel to a vector that can be used for matrix convolution

    The kernel must be time-inverted (i.e., large tau indices are short lags)
    """
    return np.asarray(strf, order='C').flatten()


def strf(nfreq, ntau, f_max, f_peak, t_peak, ampl, f_sigma, t_sigma, f_alpha, t_alpha):
    """Construct a parametric STRF

    nfreq: resolution of the filter in pixels
    ntau: time window of the filter in ms
    f_max: maximum frequency of the signal
    f_peak: center frequency for the filter
    t_peak: offset between stimulus and response in ms (range: 0 to ntau)
    ampl:   amplitude of the wavelet peak
    f_sigma: width of the filter in the frequency axis -- bigger gamma = narrower frequency band
    t_sigma: width of the filter in the time axis -- bigger sigma = narrower time band
    f_alpha: depth of inhibitory sidebands on frequency axis -- bigger f_alpha = deeper sidebands
    t_alpha: depth of inhibitory sidebands on time axis -- bigger t_alpha = deeper sidebands
    Returns the RF (nfreq, ntau), f (nfreq,), and tau (ntau,)
    """
    ntau -= 1
    scale = nfreq / 50.0
    t = np.arange(float(np.negative(ntau)), 1)
    tscale = np.arange(np.negative(ntau), 1, 2)
    x = t_peak
    f = np.arange(0, f_max + 1, float(f_max) / nfreq)
    y = f_peak
    tc = t + x
    fc = f - y
    tprime, fprime = np.meshgrid(tc, fc)
    t_sigma = t_sigma / scale
    Gtf = (ampl * np.exp(-t_sigma**2 * tprime**2 - f_sigma**2 * fprime**2) *
           (1 - t_alpha**2 * t_sigma**2 * tprime**2) * (1 - f_alpha**2 * f_sigma**2 * fprime**2))
    return Gtf, tscale, f
