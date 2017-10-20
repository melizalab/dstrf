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

    The functions returned take two optional arguments, λ and α, which control
    the L2 and L1 penalties.

    """
    from theano import function, config, shared, sparse, In
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
    spkx, spky = map(shared, spikes.nonzero())

    # regularization parameters
    reg_lambda = T.scalar('lambda')
    reg_alpha = T.scalar('alpha')
    # split out the parameter vector
    w = T.vector('w')
    dc = w[0]
    h = w[1:(nalpha + 1)]
    k = w[(nalpha + 1):]
    v = T.vector('v')
    Vx = T.dot(Xstim, k)
    # Vx has to be promoted to a matrix for structured_dot to work
    Vi = sparse.structured_dot(M, T.shape_padright(Vx))
    H = T.dot(Xspke.dimshuffle([2, 0, 1]), h).T
    mu = Vi - H - dc
    penalty = reg_lambda * T.dot(k, k) + reg_alpha * T.sqrt(T.dot(k, k) + 0.001)
    ll = T.exp(mu).sum() * dt - mu[spkx, spky].sum() + penalty
    dL = T.grad(ll, w)
    ddLv = T.grad(T.sum(dL * v), w)

    loglike = function([w, In(reg_lambda, value=0.), In(reg_alpha, value=0.)], ll)
    gradient = function([w, In(reg_lambda, value=0.), In(reg_alpha, value=0.)], dL)
    hessianv = function([w, v, In(reg_lambda, value=0.), In(reg_alpha, value=0.)], ddLv)
    return {"V": function([w], Vx),
            "V_interp": function([w], Vi),
            "lci": function([w], mu),
            "loglike": loglike,
            "gradient": gradient,
            "hessianv": hessianv}


class estimator(object):
    """Compute max-likelihood estimate of the MAT model parameters

    stim: stimulus, dimensions (nchannels, nframes)
    spikes: spike response, dimensions (nbins,)
    rf_tau: number of time lags in the kernel OR a set of temporal basis functions
    alpha_taus: the tau values for the adaptation kernel
    stim_dt: sampling rate of stimulus frames
    spike_dt: sampling rate of spike bins

    If there are multiple trials for a given stimulus, then spikes must have
    dimensions (nbins, ntrials)

    """
    def __init__(self, stim, spikes, n_rf_tau, alpha_taus, stim_dt, spike_dt):
        from theano import config
        from mat_neuron._model import adaptation
        from dstrf.strf import lagged_matrix
        self.dtype = config.floatX

        if stim.ndim == 1:
            stim = np.expand_dims(stim, 0)
        if spikes.ndim == 1:
            spikes = np.expand_dims(spikes, 1)
        nchan, nframes = stim.shape
        nbins, ntrials = spikes.shape
        self.n_spk_tau = len(alpha_taus)

        self._spikes = spikes.astype(self.dtype)
        self._X_stim = lagged_matrix(stim, n_rf_tau).astype(self.dtype)
        self._X_spike = np.zeros((nbins, self.n_spk_tau, ntrials), dtype=self.dtype)
        for i in range(ntrials):
            self._X_spike[:, :, i] = adaptation(spikes[:, i], alpha_taus, spike_dt)

        lfuns = make_likelihood(self._X_stim, self._X_spike, self._spikes, stim_dt, spike_dt)
        for k in ("V", "V_interp", "lci", "loglike", "gradient", "hessianv"):
            setattr(self, k, lfuns[k])

    def sta(self, center=False, scale=False):
        """Calculate the spike-triggered average"""
        from dstrf.strf import correlate
        spikes = self._spikes
        X = self._X_stim.copy()
        if center:
            X -= X.mean(0)
        if scale:
            X /= X.std(0)
        return correlate(X, spikes)

    def estimate(self, w0=None, reg_lambda=0, reg_alpha=0, avextol=1e-6, maxiter=300, **kwargs):
        """Compute max-likelihood estimate of the model parameters

        w0: initial guess at parameters. If not supplied (default), use STA

        Additional arguments are passed to scipy.optimize.fmin_ncg
        """
        import scipy.optimize as op
        if w0 is None:
            # we need to make sure that the initial guess is reasonably good, or
            # else the Hessian is unstable. This can happen when the stimulus is
            # not gaussian. The hacky solution is to calculate the STA on a
            # centered/scaled design matrix, and then rescale the STA so that
            # V never gets above ~100
            sta = self.sta(center=True, scale=True)
            w0 = np.r_[0, np.zeros(self.n_spk_tau, dtype=self.dtype), sta]
            Vmax = self.V(w0).max()
            if Vmax > 100:
                w0[3:] *= 90 / Vmax

        return op.fmin_ncg(self.loglike, w0, self.gradient, fhess_p=self.hessianv,
                           args=(reg_lambda, reg_alpha), avextol=avextol, maxiter=maxiter, **kwargs)
