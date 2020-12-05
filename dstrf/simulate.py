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


def whitenoise(N):
    return np.random.randn(N)


def pinknoise(N):
    """ Generate N samples of pink noise (1/f power spectrum) """
    from numpy.fft import irfft
    uneven = N % 2
    X = np.random.randn(N // 2 + 1 + uneven) + 1j * \
        np.random.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return y / np.sqrt((y ** 2).mean())


noise_fns = {
    "white": whitenoise,
    "pink": pinknoise
}


# wrappers for filters that make them config-aware. This is only really needed
# in the case of hg_dstrf because it has to update the config, but it does
# simplify dispatch a bit
def hg_dstrf(cf):
    """ Generate hg-type RF using parameters from the dstrf summary table """
    import pandas as pd
    cff = cf["data"]["filter"]
    columns = ["RF"]
    colmap = {"Latency": "t_peak",
              "Freq": "f_peak",
              "SigmaT": "t_sigma",
              "OmegaT": "t_omega",
              "SigmaF": "f_sigma",
              "OmegaF": "f_omega",
              "Pt": "Pt",
              "Amplitude": "ampl"}
    columns.extend(colmap.keys())
    #print(" - using params from STRF #{rf}".format(**cff))
    df = (pd.read_csv(cff["paramfile"], usecols=columns, index_col=[0])
            .rename(columns=colmap)
            .loc[cff["rf"]])
    # convert s to ms and Hz to kHz
    df["t_peak"] *= 1000
    df["t_sigma"] *= 1000
    df["t_omega"] /= 1000
    df["f_peak"] /= 1000
    df["f_sigma"] /= 1000
    df["f_omega"] *= 1000

    cff.update(**df)
    return filters.hg(**cff)


filter_fns = {
    "hg_dstrf": hg_dstrf
}


def get_filter(cf):
    cff = cf["data"]["filter"]
    name = cff["fn"]
    try:
        return filter_fns[name](cf)
    except KeyError:
        fn = getattr(filters, name)
        return fn(**cff)

def logistic(x,model_bounds,I_bounds,params):
        intercept = params[0]
        slope = params[1]

        E_l = model_bounds[0]
        g_l = model_bounds[1]

        L = -g_l*(E_l-I_bounds[0])
        U = -g_l*(E_l-I_bounds[1])
        exponent = np.exp(-slope*x-intercept)
        return(L+((U-L))/(1+exponent))


def multivariate_glm(cf, data, random_seed=None, trials=None):
    """Simulate GLM response to multivariate stimuli. Note: modifies data in place"""
    from .models import predict_spikes_glm
    kernel = get_filter(cf)[0]

    n_taus = len(cf.model.ataus)
    n_freq = kernel.shape[0]
    upsample = int(cf.data.dt / cf.model.dt)
    n_trials = trials or cf.data.trials
    print(" - stimulus dimension: {}". format(n_freq))
    print(" - adaptation parameters: {}". format(cf.data.adaptation))

    seed = random_seed or cf.data.trial_noise.random_seed
    print(" - seed for I_noise:", seed)
    np.random.seed(seed)
    mat.random_seed(seed)

    noise_fn = noise_fns[cf.data.trial_noise.get("color", "white")]

    for d in data:
        nchan, nframes = d["stim"].shape
        assert nchan == n_freq, "stim channels don't match"
        nbins = nframes * upsample
        spike_v = np.zeros((nbins, n_trials), dtype='i')
        spike_h = np.zeros((nbins, n_taus, n_trials), dtype='d')
        V_stim = strf.convolve(d["stim"], kernel)
        V_var = np.var(V_stim)
        spike_t = []
        for i in range(n_trials):
            V_noise = noise_fn(V_stim.size)
            snr = cf.data.trial_noise.get("snr", None)
            if snr:
                V_noise *= np.sqrt(V_var / snr / np.var(V_noise))
            elif "sd" in cf.data.trial_noise:
                V_noise *= cf.data.trial_noise.sd
            V_tot = V_stim + V_noise
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
    kernel = get_filter(cf)[0]
    n_freq = kernel.shape[0]

    n_taus = len(cf.model.ataus)
    n_freq = kernel.shape[0]
    upsample = int(cf.data.dt / cf.model.dt)
    n_trials = trials or cf.data.trials
    det_rise_time = int(cf.spike_detect.rise_dt / cf.model.dt)
    print(" - stimulus dimension: {}". format(n_freq))

    pymodel = spkc.load_model(cf.data.dynamics.model)
    if "param" in cf.data.dynamics:
        print(" - updating parameters in model:")
        for k, v in cf.data.dynamics.param.items():
            old = spkc.get_param_value(pymodel, k)
            if isinstance(v, (float, int)):
                new = v * old.units
            else:
                new = spkc.parse_quantity(v)
            spkc.set_param_value(pymodel, k, new)
            print("    + {}: {} -> {}".format(k, old, new))

    biocm_params = spkc.to_array(pymodel["parameters"])
    biocm_state0 = spkc.to_array(pymodel["state"])
    biocm_model = spkc.load_module(pymodel, os.path.dirname(cf.data.dynamics.model))
    print(" - dynamical model: {}".format(cf.data.dynamics.model))

    np.random.seed(random_seed or cf.data.trial_noise.random_seed)
    mat.random_seed(random_seed or cf.data.trial_noise.random_seed)

    noise_fn = noise_fns[cf.data.trial_noise.get("color", "white")]

    for d in data:
        nchan, nframes = d["stim"].shape
        assert nchan == n_freq, "stim channels don't match"
        nbins = nframes * upsample
        spike_v = np.zeros((nbins, n_trials), dtype='i')
        spike_h = np.zeros((nbins, n_taus, n_trials), dtype='d')
        I_stim = strf.convolve(d["stim"], kernel)
        # normalization is based on Margot's paper
        if "f_sigma" in cf.data.filter:
            I_stim *= 20 / cf.data.filter.f_sigma
        I_var = np.var(I_stim)
        if "current_recenter" in cf.data.dynamics:
            I_stim -= I_stim.mean() * cf.data.dynamics.current_recenter

        spike_t = []
        for i in range(n_trials):
            I_noise = noise_fn(I_stim.size)
            snr = cf.data.trial_noise.get("snr", None)
            if snr:
                I_noise *= np.sqrt(I_var / snr / np.var(I_noise))
            elif "sd" in cf.data.trial_noise:
                I_noise *= cf.data.trial_noise.sd
            I_tot = (I_stim + I_noise) * cf.data.dynamics.current_scaling

            #Get arguements for logistic compression function
            if "current_compression" in cf.data.dynamics :
                El_bound = spkc.get_param_value(pymodel,"E_l").magnitude
                gl_bound = spkc.get_param_value(pymodel,"g_l").magnitude

                model_bounds = (El_bound,gl_bound)
                cc = cf.data.dynamics.current_compression
                V_bounds = (cc['V_lower'],cc['V_upper'])
                comp_params = (cc['intercept'],cc['slope'])

                #Compress I_tot with logistic function
                I_tot = logistic(I_tot,model_bounds,V_bounds,comp_params)

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
