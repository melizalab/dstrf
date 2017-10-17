# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for spectrotemporal receptive fields"""
from __future__ import print_function, division, absolute_import

import numpy as np


def lagged_matrix(spec, basis):
    """Convert a (nfreq, nt) spectrogram into a design matrix

    basis: can be a positive integer specifying the number of time lags. Or it
    can be a (ntau, nbasis) matrix specifying a set of temporal basis functions
    spanning ntau time lags (for example, the output of cosbasis).

    The output is an (nt, nfreq * nbasis) array. (nbasis = basis when basis is
    an integer)

    """
    from scipy.linalg import hankel
    if spec.ndim == 1:
        spec = np.expand_dims(spec, 0)
    nf, nt = spec.shape
    if np.isscalar(basis):
        ntau = nbasis = basis
    else:
        ntau, nbasis = basis.shape
    X = np.zeros((nt, nf * nbasis), dtype=spec.dtype)
    padding = np.zeros(ntau - 1, dtype=spec.dtype)
    for i in range(nf):
        h = hankel(np.concatenate([padding, spec[i, :(-ntau + 1)]]), spec[i, -ntau:])
        if not np.isscalar(basis):
            h = np.dot(h, basis)
        X[:, (i * nbasis):((i + 1) * nbasis)] = h
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
    return np.arccos(cos_theta)


def cosbasis(nt, nb, peaks=None, lin=10):
    """Make a nonlinearly stretched basis consisting of raised cosines

    nt:    number of time points
    nb:    number of basis vectors
    peaks: 2-element sequence containing locations of first and last peaks
    lin:   offset for nonlinear stretching of x axis (larger values -> more linear spacing)
    """
    from numpy import cos, clip, pi
    if peaks is None:
        peaks = np.asarray([0, nt * (1 - 1.5 / nb)])

    def nlin(x):
        # nonlinearity for stretching x axis
        return np.log(x + 1e-20)

    y = nlin(peaks + lin)                     # nonlinear transformed first and last
    db = (y[1] - y[0]) / (nb - 1)             # spacing between peaks
    ctrs = np.arange(y[0], y[1] + db, db)     # centers of peaks
    mxt = np.exp(y[1] + 2 * db) - 1e-20 - lin       # maximum time bin
    kt0 = np.arange(0, mxt)
    nt0 = len(kt0)

    def cbas(c):
        return (cos(clip((nlin(kt0 + lin) - c) * pi / db / 2, -pi, pi)) + 1) / 2

    basis = np.column_stack([cbas(c)[::-1] for c in ctrs[::-1]])
    # pad/crop
    if nt0 > nt:
        basis = basis[-nt:]
    elif nt0 < nt:
        basis = np.r_[np.zeros((nt - nt0, nb)), basis]
    # normalize to unit vectors
    basis /= np.linalg.norm(basis, axis=0)
    return basis


def to_basis(v, basis):
    """Find best projection of v into basis"""
    return np.dot(v, np.linalg.pinv(basis).T)


def from_basis(v, basis):
    """Calculate projection of v from basis to unit vector space"""
    return np.dot(basis, v.T).T
