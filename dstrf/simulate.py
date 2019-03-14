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


def multivariate_glm(cf, data, random_seed=None, trials=None):
    """Simulate GLM response to multivariate stimuli. Note: modifies data in place"""
    from .models import predict_spikes_glm
    filter_params = cf.data.filter.copy()
    filter_fn = getattr(filters, filter_params.pop("fn"))
    kernel = filter_fn(**filter_params)[0]

    n_taus = len(cf.model.ataus)
    n_freq = kernel.shape[0]
    upsample = int(cf.data.dt / cf.model.dt)
    n_trials = trials or cf.data.trials
    print(" - stimulus dimension: {}". format(n_freq))
    print(" - adaptation parameters: {}". format(cf.data.adaptation))

    np.random.seed(random_seed or cf.data.random_seed)
    mat.random_seed(random_seed or cf.data.random_seed)

    for d in data:
        nchan, nframes = d["stim"].shape
        assert nchan == n_freq, "stim channels don't match"
        nbins = nframes * upsample
        spike_v = np.zeros((nbins, n_trials), dtype='i')
        spike_h = np.zeros((nbins, n_taus, n_trials), dtype='d')
        V_stim = strf.convolve(d["stim"], kernel)
        spike_t = []
        for i in range(n_trials):
            V_tot = V_stim + np.random.randn(V_stim.size) * cf.data.trial_noise.sd
            spikes = predict_spikes_glm(V_tot, cf.data.adaptation, cf)
            spike_v[:, i] = spikes
            spike_h[:, :, i] = mat.adaptation(spikes, cf.model.ataus, cf.model.dt)
            spike_t.append(spikes.nonzero()[0])
        d["spike_v"] = spike_v
        d["spike_h"] = spike_h
        d["spike_t"] = spike_t
        d["spike_dt"] = cf.model.dt
        d["V"] = V_tot

    return data


def multivariate_dynamical(cf, data, random_seed=None, trials=None):
    """Univariate white noise stimulus, biophysical model """
    filter_params = cf.data.filter.copy()
    filter_fn = getattr(filters, filter_params.pop("fn"))
    kernel = filter_fn(**filter_params)[0]
    n_freq = kernel.shape[0]

    n_taus = len(cf.model.ataus)
    n_freq = kernel.shape[0]
    upsample = int(cf.data.dt / cf.model.dt)
    n_trials = trials or cf.data.trials
    det_rise_time = int(cf.spike_detect.rise_dt / cf.model.dt)
    print(" - stimulus dimension: {}". format(n_freq))

    pymodel = spkc.load_model(cf.data.dynamics.model)
    biocm_params = spkc.to_array(pymodel["parameters"])
    biocm_state0 = spkc.to_array(pymodel["state"])
    biocm_model = spkc.load_module(pymodel, os.path.dirname(cf.data.dynamics.model))
    print(" - dynamical model: {}". format(cf.data.dynamics.model))

    np.random.seed(random_seed or cf.data.random_seed)
    mat.random_seed(random_seed or cf.data.random_seed)

    for d in data:
        nchan, nframes = d["stim"].shape
        assert nchan == n_freq, "stim channels don't match"
        nbins = nframes * upsample
        spike_v = np.zeros((nbins, n_trials), dtype='i')
        spike_h = np.zeros((nbins, n_taus, n_trials), dtype='d')
        I_stim = strf.convolve(d["stim"], kernel)
        spike_t = []
        for i in range(n_trials):
            I_noise = np.random.randn(I_stim.size) * cf.data.trial_noise.sd
            I_tot = (I_stim + I_noise) * cf.data.dynamics.current_scaling
            X = biocm_model.integrate(biocm_params, biocm_state0, I_tot, cf.data.dt, cf.model.dt)
            det = qs.detector(cf.spike_detect.thresh, det_rise_time)
            V = X[:, 0]
            spike_times = det(V)
            spike_array = np.zeros(V.size, 'i')
            spike_array[spike_times] = 1
            H = mat.adaptation(spike_array, cf.model.ataus, cf.model.dt)
            spike_v[:, i] = spike_array
            spike_h[:, :, i] = H
            spike_t.append(spike_times)
        d["spike_v"] = spike_v
        d["spike_h"] = spike_h
        d["spike_t"] = spike_t
        d["spike_dt"] = cf.model.dt
        d["I"] = I_tot
        d["V"] = V
        d["state"] = X

    return data
