# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for ML estimation of MAT parameters"""

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
    from theano import function, shared, sparse, gradient
    import theano.tensor as T
    import scipy.sparse as sps

    nframes, nk = stim_design.shape
    nbins, nalpha = spike_design.shape
    upsample = stim_dt // spike_dt
    if upsample =! (nbins // nframes):
        raise ValueError("size of design matrices does not match sampling rates")
    # make a design matrix for the stimulus
    # X_stim = design_matrix(stim, n_rf_tau)
    # make an interpolation matrix
    interp = sps.kron(sps.eye(nframes),
                      np.ones((upsample, 1), dtype='i'),
                      format='csc')

    # convolve the spike train with the exponential basis set
    # H = np.column_stack([mat.predict_adaptation(spikes, ta, model_dt) for ta in tau_alpha])

    M = shared(interp)
    dt = shared(spike_dt)
    Xstim = shared(X_stim)
    Xspke = shared(X_spike)
    sidx = shared(spk)

    # split out the parameter vector
    w = T.dvector('w')
    dc = w[0]
    alpha = w[1:(nalpha+1)]
    k = w[(nalpha+1):]
    v = T.dvector('v')
    Vx = T.dot(Xstim, k)
    Hx = T.dot(Xspke, alpha)
    Vi = sparse.structured_dot(M, Vx.reshape((Vx.size, 1))).squeeze() - Hx - dc
    ll = T.exp(Vi).sum() * dt - Vi[sidx].sum()
    dL = T.grad(ll, w)
    ddL = gradient.hessian(ll, w)

    return {"lci": function([w], Vi),
            "loglike": function([w], ll),
            "gradient": function([w], dL),
            "hessian": function([w], ddL)}
