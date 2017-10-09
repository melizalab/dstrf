# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for ML estimation of MAT parameters"""
from __future__ import print_function, division, absolute_import

import numpy as np


def make_likelihood(stim_design, spike_design, spikes, stim_dt, spike_dt):
    """Generate functions for evaluating negative log-likelihood of model

    stim_design: design matrix for the stimulus (nframes x nk)
    spike_design: design matrix for the spike history terms (nbins x nα)
    spikes: spike array, dimension (nbins,)
    stim_dt: sampling rate of stimulus frames
    spike_dt: sampling rate of spike bins

    If there are multiple trials for a given stimulus, then spike_design becomes
    (nbins, nα, ntrials) and spikes becomes (nbins, ntrials)

    Returns a dictionary with several functions (lci = log conditional
    intensity, loglike = negative log likelihood, gradient, hessian) and shared
    variables (X_stim, X_stim, spikes). The functions take a single argument,
    the parameters as a vector. The ordering of the parameters is as follows: ω,
    α1 ... αN, k1 ... kN. The shared variables can be used to alter the design
    matrices in place, but it's often simpler just to call this function again.

    NB: this function will throw a rather verbose warning about an optimization
    failure; however, the returned functions will work just fine.

    """
    from theano import function, config, shared, sparse, gradient
    import theano.tensor as T
    import scipy.sparse as sps

    if spike_design.ndim == 2:
        spike_design = np.expand_dims(spike_design, 2)
    if spikes.ndim == 1:
        spikes = np.expand_dims(spikes, 1)

    nframes, nk = stim_design.shape
    nbins, nalpha, ntrials = spike_design.shape
    upsample = int(stim_dt / spike_dt)
    if upsample != (nbins // nframes):
        raise ValueError("size of design matrices does not match sampling rates")
    if spikes.shape != (nbins, ntrials):
        raise ValueError("size of spikes matrix does not design matrix")
    # make an interpolation matrix
    interp = sps.kron(sps.eye(nframes),
                      np.ones((upsample, 1), dtype=config.floatX),
                      format='csc')

    M = shared(interp)
    dt = shared(spike_dt)
    Xstim = shared(stim_design)
    Xspke = shared(spike_design)
    spk = shared(spikes)

    # split out the parameter vector
    w = T.vector('w')
    dc = w[0]
    alpha = w[1:(nalpha+1)]
    k = w[(nalpha+1):]
    v = T.vector('v')
    Vx = T.dot(Xstim, k)
    # Vx has to be promoted to a matrix for structured_dot to work
    Vi = sparse.structured_dot(M, T.shape_padright(Vx))
    H = T.dot(Xspke.dimshuffle([2, 0 , 1]), alpha).T
    mu = Vi - H - dc
    ll = T.exp(mu).sum() * dt - mu[spk.nonzero()].sum()
    dL = T.grad(ll, w)
    ddL = gradient.hessian(ll, w)

    return {"X_stim": Xstim, "X_spike": Xspke, "spikes": spk,
            "lci": function([w], mu),
            "loglike": function([w], ll),
            "gradient": function([w], dL),
            "hessian": function([w], ddL)
    }


def estimate(stim, spikes, n_rf_tau, alpha_taus, stim_dt, spike_dt, w0=None, dry_run=False, **kwargs):
    """Compute max-likelihood estimate of the MAT model parameters

    stim: stimulus, dimensions (nchannels, nframes)
    spikes: spike response, dimensions (nbins,)
    n_rf_tau: number of time lags in the kernel
    alpha_taus: the tau values for the adaptation kernel
    stim_dt: sampling rate of stimulus frames
    spike_dt: sampling rate of spike bins
    w0: initial guess at parameters (optional)
    dry_run: if True, return likelihood functions but don't run the optimization

    Additional arguments are passed to scipy.optimize.fmin_ncg

    If there are multiple trials for a given stimulus, then spikes becomes
    (nbins, ntrials)

    Returns parameter estimates

    """
    from theano import config
    from dstrf.mat import adaptation
    from dstrf.strf import lagged_matrix, correlate
    import scipy.optimize as op

    if stim.ndim == 1:
        stim = np.expand_dims(stim, 0)
    if spikes.ndim == 1:
        spikes = np.expand_dims(spikes, 1)

    nchan, nframes = stim.shape
    nbins, ntrials = spikes.shape
    n_spk_tau = len(alpha_taus)

    if w0 is not None and  w0.size != (1 + n_spk_tau + n_rf_tau):
        raise ValueError("w0 needs to be a vector with size {}".format(1 + n_spk_tau + n_rf_tau))

    X_stim = lagged_matrix(stim, n_rf_tau)
    X_spike = np.zeros((nbins, n_spk_tau, ntrials), dtype=config.floatX)
    for i in range(ntrials):
        for j, tau in enumerate(alpha_taus):
            X_spike[:, j, i] = adaptation(spikes[:, i], tau, spike_dt)

    if w0 is None:
        sta = correlate(X_stim, spikes)
        w0 = np.r_[0, np.zeros(n_spk_tau), sta]

    lfuns = make_likelihood(X_stim, X_spike, spikes, stim_dt, spike_dt)
    if dry_run:
        return lfuns

    maxiter = kwargs.pop("maxiter", 100)
    return op.fmin_ncg(lfuns['loglike'], w0, lfuns['gradient'],
                     fhess=lfuns['hessian'], maxiter=maxiter, **kwargs)
