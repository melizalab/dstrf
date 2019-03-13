# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" This module will simulate responses of various models to various stimuli """
from __future__ import print_function, division

import os
import numpy as np
import quickspikes as qs

import mat_neuron._model as mat
from dstrf import filters, strf
import spyks.core as spkc


def multivariate_noise_glm(cf, random_seed=None, trials=None):
    """Multivariate white noise stimulus, glm model"""
    from .models import predict_spikes_glm
    stim_dt = cf.data.dt
    filter_params = cf.data.filter.copy()
    filter_fn = getattr(filters, filter_params.pop("fn"))
    kernel, _, _ = filter_fn(**filter_params)

    n_freq = kernel.shape[0]
    n_bins = int(cf.data.duration / cf.model.dt)
    upsample = int(stim_dt / cf.model.dt)
    n_frames = n_bins // upsample
    n_trials = trials or cf.data.trials

    stim = np.random.randn(n_freq, n_frames)
    stim[:, :100] = 0

    np.random.seed(random_seed or cf.data.random_seed)
    mat.random_seed(random_seed or cf.data.random_seed)
    data = []
    V_stim = strf.convolve(stim, kernel)
    for i in range(n_trials):
        V_tot = V_stim + np.random.randn(V_stim.size) * cf.data.trial_noise.sd
        spike_v = predict_spikes_glm(V_tot, cf.data.adaptation, cf)
        H = mat.adaptation(spike_v, cf.model.ataus, cf.model.dt)
        data.append({
            "stim": stim,
            "V": V_tot,
            "H": H,
            "duration": cf.data.duration,
            "spike_t": np.nonzero(spike_v)[0],
            "spike_v": spike_v
        })
    return data


def univariate_noise_dynamical(cf, random_seed=None, trials=None):
    """Univariate white noise stimulus, biophysical model """
    stim_dt = cf.data.dt
    ntau = cf.model.filter.len
    kernel, _ = filters.gammadiff(ntau * stim_dt / cf.data.filter.taus[0],
                                  ntau * stim_dt / cf.data.filter.taus[1],
                                  cf.data.filter.amplitude,
                                  ntau * stim_dt, stim_dt)

    n_bins = int(cf.data.duration / cf.model.dt)
    upsample = int(cf.data.dt / cf.model.dt)
    n_frames = n_bins // upsample
    det_rise_time = int(cf.spike_detect.rise_dt / cf.model.dt)

    stim = np.random.randn(n_frames)
    stim[:100] = 0

    pymodel = spkc.load_model(cf.data.dynamics.model)
    biocm_params = spkc.to_array(pymodel["parameters"])
    biocm_state0 = spkc.to_array(pymodel["state"])
    biocm_model = spkc.load_module(pymodel, os.path.dirname(cf.data.dynamics.model))

    np.random.seed(random_seed or cf.data.random_seed)
    mat.random_seed(random_seed or cf.data.random_seed)
    data = []
    I_stim = np.convolve(stim, kernel, mode="full")[:stim.size]
    for i in range(trials or cf.data.trials):
        I_noise = np.random.randn(I_stim.size) * cf.data.trial_noise.sd
        I_tot = (I_stim + I_noise) * cf.data.dynamics.current_scaling
        X = biocm_model.integrate(biocm_params, biocm_state0, I_tot, cf.data.dt, cf.model.dt)
        det = qs.detector(cf.spike_detect.thresh, det_rise_time)
        V = X[:, 0]
        spike_t = det(V)
        spike_v = np.zeros(V.size, 'i')
        spike_v[spike_t] = 1
        H = mat.adaptation(spike_v, cf.model.ataus, cf.model.dt)
        data.append({
            "stim": stim,
            "I": I_tot,
            "state": X,
            "duration": cf.data.duration,
            "spike_t": np.asarray(spike_t),
            "spike_v": spike_v,
            "H": H
        })

    return data
