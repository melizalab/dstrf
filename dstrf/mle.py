# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for ML estimation of MAT parameters"""
from __future__ import print_function, division, absolute_import

import numpy as np


def make_likelihood(stim_design, spike_design, spikes, stim_dt, spike_dt):
    """Generate functions for evaluating negative log-likelihood of model

    stim_design: design matrix for the stimulus (nframes x nk)
    spike_design: design matrix for the spike history terms (nbins x nα)
    spikes: vector of spike time indi
    stim_dt: sampling rate of stimulus frames
    spike_dt: sampling rate of spike bins

    Returns a tuple of functions: (loglikelihood, gradient, hessian,). These
    functions take a single argument, the parameters as a vector. The ordering
    of the parameters is as follows: ω, α1 ... αN, k1 ... kN

    """
    from theano import function, config, shared, sparse, gradient
    import theano.tensor as T
    import scipy.sparse as sps

    nframes, nk = stim_design.shape
    nbins, nalpha = spike_design.shape
    upsample = int(stim_dt / spike_dt)
    if upsample != (nbins // nframes):
        raise ValueError("size of design matrices does not match sampling rates")
    # make an interpolation matrix
    interp = sps.kron(sps.eye(nframes),
                      np.ones((upsample, 1), dtype=config.floatX),
                      format='csc')

    M = shared(interp)
    dt = shared(spike_dt)
    Xstim = shared(stim_design)
    Xspke = shared(spike_design)
    sidx = shared(spikes)

    # split out the parameter vector
    w = T.dvector('w')
    dc = w[0]
    alpha = w[1:(nalpha+1)]
    k = w[(nalpha+1):]
    v = T.dvector('v')
    Vx = T.dot(Xstim, k)
    Hx = T.dot(Xspke, alpha)
    Vi = sparse.structured_dot(M, Vx.reshape((Vx.size, 1))).squeeze() - Hx - dc
    # if spikes.size == spike_design.shape[1]:
    #     ll = T.exp(Vi).sum() * dt - (Vi * sidx).sum()
    # else:
    ll = T.exp(Vi).sum() * dt - Vi[sidx].sum()
    dL = T.grad(ll, w)
    ddL = gradient.hessian(ll, w)

    return {"lci": function([w], Vi),
            "loglike": function([w], ll),
            "gradient": function([w], dL),
            "hessian": function([w], ddL)}


def estimate(stim, spikes, n_rf_tau, alpha_taus, stim_dt, spike_dt, w0=None, **kwargs):
    """Compute max-likelihood estimate of the MAT model parameters

    stim: stimulus, dimensions (nchannels, nframes)
    spikes: spike response, dimensions (nbins,)

    """
    from dstrf.mat import adaptation
    from dstrf.strf import lagged_matrix, correlate
    import scipy.optimize as op

    nchan, nframes = stim.shape
    nbins, = spikes.shape

    X_stim = lagged_matrix(stim, n_rf_tau)
    X_spike = np.column_stack([adaptation(spikes, tau, spike_dt) for tau in alpha_taus])
    n_spk_tau = X_spike.shape[1]

    if w0 is None:
        sta = correlate(X_stim, spikes)
        w0 = np.r_[0, np.zeros(n_spk_tau), sta]
    elif w0.size != (1 + n_spk_tau + n_rf_tau):
        raise ValueError("w0 needs to be a vector with size {}".format(1 + n_spk_tau + n_rf_tau))

    spike_t = spikes.nonzero()[0]
    lfuns = make_likelihood(X_stim, X_spike, spike_t, stim_dt, model_dt)

    maxiter = kwargs.pop(maxiter, 100)
    return op.fmin_ncg(lfuns['loglike'], w, lfuns['gradient'],
                       fhess=lfuns['hessian'], maxiter=maxiter, **kwargs)
