# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for spectrotemporal receptive fields"""
from __future__ import print_function, division, absolute_import

import numpy as np


def convolve(spec, strf):
    """Convolve (nfreq, nt) spectrogram with a (nfreq, ntau) spectrotemporal kernel.

    This function uses the numpy convolve function, but the kernel should be
    flipped (large indices are short lags) for compatibility with lagged_matrix.

    """
    if spec.ndim == 1:
        spec = np.expand_dims(spec, 0)
    nf, nt = spec.shape
    X = np.zeros(nt)
    for i in range(nf):
        X += np.correlate(spec[i], strf[i], mode="full")[:nt]
    return X.squeeze()


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
        h = hankel(np.concatenate([padding, spec[i, : (-ntau + 1)]]), spec[i, -ntau:])
        if not np.isscalar(basis):
            h = np.dot(h, basis)
        X[:, (i * nbasis) : ((i + 1) * nbasis)] = h
    return X


def as_vector(strf):
    """Convert a (nfreq, ntau) kernel to a vector that can be used for matrix convolution

    To be compatible with the output of lagged_matrix, the kernel must be
    time-inverted (i.e., large tau indices are short lags)

    """
    return np.ravel(strf, order="C")


def as_matrix(k, basis):
    """Convert an (nfreq * nbasis,) kernel k back into an (nfreq, ntau) strf form

    As with lagged_matrix, if 'basis' argument is a positive integer, it's
    interpreted as the number of time lags. If it's an (ntau, nbasis) matrix,
    then it's interpreted as a set of temporal basis functions, and the RF
    is projected back into the standard discrete time basis.

    """
    if np.isscalar(basis):
        ntau = nbasis = basis
    else:
        ntau, nbasis = basis.shape
    nfreq = k.size // nbasis
    rf = k.reshape(nfreq, nbasis)
    if np.isscalar(basis):
        return rf
    else:
        return from_basis(rf, basis)


def correlate(stim_design, spikes):
    """Correlation between stim (as design matrix) and spikes (i.e., spike-triggered average)

    NB: divide by variance of stimulus (or multiply by inverse covariance) to recover filter"""
    if spikes.ndim == 1:
        spikes = np.expand_dims(spikes, 1)

    nframes, nfeat = stim_design.shape
    nbins, ntrials = spikes.shape
    # coarse binning of stimulus
    psth = np.sum(spikes.reshape(nframes, ntrials, -1), axis=(1, 2))
    return np.dot(stim_design.T, psth) / psth.sum()


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

    y = nlin(peaks + lin)  # nonlinear transformed first and last
    db = (y[1] - y[0]) / (nb - 1)  # spacing between peaks
    ctrs = np.arange(y[0], y[1] + db, db)  # centers of peaks
    mxt = np.exp(y[1] + 2 * db) - 1e-20 - lin  # maximum time bin
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
    """Find best projection of filter v into basis"""
    return np.dot(v, np.linalg.pinv(basis).T)


def from_basis(v, basis):
    """Calculate projection of filter v from basis to unit vector space"""
    if np.isscalar(basis):
        return v
    else:
        return np.dot(basis, v.T).T


def factorize(k, rank=1, thresh=None):
    """Compute low-rank approximation of (nf, nt) filter k

    Returns k_f (nf, rank), k_t (rank, nt) such that np.dot(k_f, k_t) ~= k.

    If thresh is None, the rank is as specified by the rank parameter. If thresh
    is not None, the rank will be equal to the number of eigenvalues of k that
    are greater than thresh, or rank, whichever is larger.

    """
    U, s, V = np.linalg.svd(k)
    if thresh is not None:
        rank = max(rank, sum(s > thresh))
    # TODO: flip signs if shape is opposite to input
    return (
        U[:, :rank],
        V[:rank] * s[:rank, np.newaxis],
    )


def unpack_factors(v, nfreq, rank=1):
    """Unpack factorized RF parameters into componenent matrices

    v - the factorized RF parameters as a 1D array with dimension (nfreq * rank
    + ntau * rank). This vector can be produced by
    np.concatenate([np.flatten(v) for v in factorize(RF, rank)])

    Returns k_f (nf, rank) and k_t (rank, nt) such that np.dot(k_f, k_t) ~= k
    """
    nv = v.size
    nfv = nfreq * rank
    ntv = nv - nfv
    ntau = int(ntv / rank)
    assert (
        nv == (nfreq + ntau) * rank
    ), "dimensions of factorized/flattened RF don't match nfreq and rank"

    k_f = v[:nfv].reshape((nfreq, rank))
    k_t = v[nfv:].reshape((rank, ntau))
    return k_f, k_t


def defactorize(v, nfreq, rank=1):
    """Convert a factorized RF into the full-rank matrix

    v - the factorized RF parameters as a 1D array with dimension (nfreq * rank
    + ntau * rank). This vector can be produced by
    np.concatenate([np.flatten(v) for v in factorize(RF, rank)])

    """
    return np.dot(*unpack_factors(v, nfreq, rank))
