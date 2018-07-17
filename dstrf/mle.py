# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for ML estimation of MAT parameters"""
from __future__ import print_function, division, absolute_import
import numpy as np


class mat(object):
    """Compute max-likelihood estimate of the MAT model parameters

    stim: stimulus, dimensions (nchannels, nframes)
    rf_tau: number of time lags in the kernel OR a set of temporal basis functions
    spike_v: spike response, dimensions (nbins, [ntrials])
    spike_h: spike history (i.e. spike_v convolved with basis kernels), dim (nbins, nbasis, [ntrials])
    stim_dt: sampling rate of stimulus frames
    spike_dt: sampling rate of spike bins
    nlin: the nonlinearity. Allowed values: "exp" (default), "softplus", "sigmoid"

    If there are multiple trials for a given stimulus, then spikes must have
    dimensions (nbins, ntrials)

    """
    def __init__(self, stim, rf_tau, spike_v, spike_h, stim_dt, spike_dt, nlin="exp"):
        from theano import config
        import scipy.sparse as sps
        from dstrf.strf import lagged_matrix
        self.dtype = config.floatX

        if stim.ndim == 1:
            stim = np.expand_dims(stim, 0)
        if spike_v.ndim == 1:
            spike_v = np.expand_dims(spike_v, 1)
        if spike_h.ndim == 2:
            spike_h = np.expand_dims(spike_h, 2)

        self._nchannels, nframes = stim.shape
        nbins, ntau, ntrials = spike_h.shape
        upsample = int(stim_dt / spike_dt)
        if upsample != (nbins // nframes):
            raise ValueError("size of design matrices does not match sampling rates")
        if spike_v.shape != (nbins, ntrials):
            raise ValueError("size of spikes matrix does not design matrix")

        self._spike_dt = spike_dt
        self._nlin = nlin
        self._spikes = sps.csc_matrix(spike_v)
        self._X_stim = lagged_matrix(stim, rf_tau).astype(self.dtype)
        self._X_spike = spike_h.astype(self.dtype)
        self._interp = sps.kron(sps.eye(nframes),
                                np.ones((upsample, 1), dtype=config.floatX),
                                format='csc')
        self.select_data()
        self._make_functions()

    @property
    def X_stim(self):
        return self._X_stim

    @property
    def X_spike(self):
        return self._X_spike

    def _nlin_theano(self, mu):
        import theano.tensor as T
        from theano.tensor import nnet
        if self._nlin == "exp":
            return T.exp(mu)
        elif self._nlin == "softplus":
            return nnet.softplus(mu)
        elif self._nlin == "sigmoid":
            return nnet.sigmoid(mu)
        else:
            raise ValueError("unknown nonlinearity type: {}".format(self._nlin))

    def _make_functions(self):
        """Generate the theano graph"""
        from theano import function, sparse, In
        import theano.tensor as T

        nalpha = self._X_spike.shape[1]

        # regularization parameters
        reg_lambda = T.scalar('lambda')
        reg_alpha = T.scalar('alpha')
        # split out the parameter vector
        w = T.vector('w')
        dc = w[0]
        h = w[1:(nalpha + 1)]
        k = w[(nalpha + 1):]
        # elastic net penalty
        penalty = reg_lambda * T.dot(k, k) + reg_alpha * T.sqrt(T.dot(k, k) + 0.001)
        # Vx has to be promoted to a matrix for structured_dot to work
        Vx = T.dot(self._sh_X_stim, k)
        Vi = sparse.structured_dot(self._sh_interp, T.shape_padright(Vx))
        H = T.dot(self._sh_X_spike.dimshuffle([2, 0, 1]), h).T
        mu = Vi - H - dc
        lmb = self._nlin_theano(mu)
        # this version of the log-likelihood is faster, but the gradient doesn't work
        llf = lmb.sum() * self._spike_dt - sparse.sp_sum(sparse.structured_log(self._sh_Y_spike * lmb), sparse_grad=True) + penalty
        # this version has a working gradient
        ll = lmb.sum() * self._spike_dt - sparse.sp_sum(self._sh_Y_spike * T.log(lmb), sparse_grad=True) + penalty
        dL = T.grad(ll, w)
        v = T.vector('v')
        ddLv = T.grad(T.sum(dL * v), w)

        self.V = function([w], Vx)
        self.V_interp = function([w], Vi)
        self.lci = function([w], mu)
        self.loglike = function([w, In(reg_lambda, value=0.), In(reg_alpha, value=0.)], llf)
        self.gradient = function([w, In(reg_lambda, value=0.), In(reg_alpha, value=0.)], dL)
        self.hessianv = function([w, v, In(reg_lambda, value=0.), In(reg_alpha, value=0.)], ddLv)

    def select_data(self, fselect=None, bselect=None):
        """Updates the shared variables containing the data to be fit

        Default behavior is to populate the shared variables with the data
        provided to the constructor. However, if fselect (selecting frames) and
        bselect (selecting bins) are specified, then only a subset of data is
        used. This is useful for cross-validation.

        """
        from theano import shared
        if fselect is None or bselect is None:
            self._sh_interp = shared(self._interp)
            self._sh_X_stim = shared(self._X_stim)
            self._sh_X_spike = shared(self._X_spike)
            self._sh_Y_spike = shared(self._spikes)
        else:
            self._sh_interp.set_value(self._interp[bselect][:, fselect])
            self._sh_X_stim.set_value(self._X_stim[fselect])
            self._sh_X_spike.set_value(self._X_spike[bselect])
            self._sh_Y_spike.set_value(self._spikes[bselect])

    def sta(self, center=False, scale=False):
        """Calculate the spike-triggered average"""
        from dstrf.strf import correlate
        spikes = self._spikes.toarray()
        X = self._X_stim.copy()
        if center:
            X -= X.mean(0)
        if scale:
            X /= X.std(0)
        return correlate(X, spikes)

    def param0(self):
        """Returns a parameter vector with a good starting guess"""
        kdim = self._X_stim.shape[1]
        nbins, hdim, ntrials = self._X_spike.shape
        meanrate = self._spikes.sum(0).mean() / nbins
        return np.r_[np.exp(meanrate),
                     np.zeros(hdim + kdim)].astype(self.dtype)


    def estimate(self, w0=None, reg_lambda=0, reg_alpha=0, avextol=1e-6, maxiter=300, **kwargs):

        """Compute max-likelihood estimate of the model parameters

        w0: initial guess at parameters. If not supplied (default), sets omega
        to the mean firing rate and all other parameters to zero.

        Additional arguments are passed to scipy.optimize.fmin_ncg

        """
        import scipy.optimize as op
        if w0 is None:
            w0 = self.param0()

        return op.fmin_ncg(self.loglike, w0, self.gradient, fhess_p=self.hessianv,
                           args=(reg_lambda, reg_alpha), avextol=avextol, maxiter=maxiter, **kwargs)

    def predict(self, w0, tau_params, V=None, random_state=None):
        """Generate a predicted spike train

        w0: the estimator's parameters
        tau_params: - the values for t1, t2,...tN and tref (which are not estimated)
        V:  if specified, skips calculating the convolution of the stim with the kernel

        """
        import mat_neuron._model as mat
        if random_state is not None:
            mat.random_seed(random_state)
        nbins, hdim, ntrials = self._X_spike.shape
        omega = w0[0]
        hvalues = w0[1:(1 + hdim)]
        tvalues = tau_params[:hdim]
        tref = tau_params[-1]
        if self._nlin == "exp":
            f = mat.predict_poisson
        elif self._nlin == "softplus":
            f = mat.predict_softplus
        if V is None:
            V = self.V(w0)
        return f(V - omega, hvalues, tvalues, tref, self._spike_dt, nbins // V.size)


class matfact(mat):
    """ The MAT dSTRF model, but with factorized (bilinear) kernel

    stim: stimulus, dimensions (nchannels, nframes)
    rf_tau: number of time lags in the kernel OR a set of temporal basis functions
    rf_rank: the rank of the factorized rf (1 or 2 is usually good)
    spike_v: spike response, dimensions (nbins, [ntrials])
    spike_h: spike history (i.e. spike_v convolved with basis kernels), dim (nbins, nbasis, [ntrials])
    stim_dt: sampling rate of stimulus frames
    spike_dt: sampling rate of spike bins
    nlin: the nonlinearity. Allowed values: "exp" (default), "softplus", "sigmoid"

    If there are multiple trials for a given stimulus, then spikes must have
    dimensions (nbins, ntrials)

    """
    def __init__(self, stim, rf_tau, rf_rank, spike_v, spike_h, stim_dt, spike_dt, nlin="exp"):
        self._rank = rf_rank
        super(matfact, self).__init__(stim, rf_tau, spike_v, spike_h, stim_dt, spike_dt, nlin)

    def _make_functions(self):
        """Generate the theano graph"""
        from theano import function, sparse, In
        import theano.tensor as T

        nframes, nk = self._X_stim.shape
        nbins, nalpha, ntrials = self._X_spike.shape
        nt = nk // self._nchannels
        nkf = self._nchannels * self._rank
        nkt = nt * self._rank

        # regularization parameters
        reg_lambda = T.scalar('lambda')
        reg_alpha = T.scalar('alpha')
        # split out the parameter vector
        w = T.vector('w')
        dc = w[0]
        h = w[1:(nalpha + 1)]
        kf = w[(nalpha + 1):(nalpha + nkf + 1)]
        kt = w[(nalpha + nkf + 1):(nalpha + nkf + nkt + 1)]
        k = T.dot(kf.reshape((self._nchannels, self._rank)), kt.reshape((self._rank, nt))).ravel()
        # elastic net penalty
        penalty = reg_lambda * T.dot(k, k) + reg_alpha * T.sqrt(T.dot(k, k) + 0.001)
        # Vx has to be promoted to a matrix for structured_dot to work
        Vx = T.dot(self._sh_X_stim, k)
        Vi = sparse.structured_dot(self._sh_interp, T.shape_padright(Vx))
        H = T.dot(self._sh_X_spike.dimshuffle([2, 0, 1]), h).T
        mu = Vi - H - dc
        lmb = self._nlin_theano(mu)
        # this version of the log-likelihood is faster, but the gradient doesn't work
        llf = lmb.sum() * self._spike_dt - sparse.sp_sum(sparse.structured_log(self._sh_Y_spike * lmb), sparse_grad=True) + penalty
        # this version has a working gradient
        ll = lmb.sum() * self._spike_dt - sparse.sp_sum(self._sh_Y_spike * T.log(lmb), sparse_grad=True) + penalty
        dL = T.grad(ll, w)
        v = T.vector('v')
        ddLv = T.grad(T.sum(dL * v), w)

        self.V = function([w], Vx)
        self.V_interp = function([w], Vi)
        self.lci = function([w], mu)
        self.loglike = function([w, In(reg_lambda, value=0.), In(reg_alpha, value=0.)], llf)
        self.gradient = function([w, In(reg_lambda, value=0.), In(reg_alpha, value=0.)], dL)
        self.hessianv = function([w, v, In(reg_lambda, value=0.), In(reg_alpha, value=0.)], ddLv)

    def param0(self, random_seed=10, random_sd=0.1):
        """Returns a parameter vector with a good starting guess

        For the factorized model, the RF *cannot* be all zeros because of the
        sign ambiguity in the factorization. So instead we use normally
        distributed random numbers with a fixed seed.

        """
        from numpy import random
        random.seed(random_seed)
        nframes, nk = self._X_stim.shape
        nkf = self._nchannels * self._rank
        nkt = nk // self._nchannels * self._rank
        kdim = nkt + nkf
        nbins, hdim, ntrials = self._X_spike.shape
        meanrate = self._spikes.sum(0).mean() / nbins
        return np.r_[np.exp(meanrate),
                     np.zeros(hdim),
                     random_sd * random.randn(kdim)].astype(self.dtype)

    def strf(self, w):
        """Returns the full-rank RF"""
        from dstrf.strf import defactorize
        return defactorize(w[3:], self._nchannels, self._rank)
