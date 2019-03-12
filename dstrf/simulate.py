# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" This module will simulate responses of various models to various stimuli """
from __future__ import print_function, division

import os
import numpy as np
import quickspikes as qs

import mat_neuron._model as mat
from dstrf import filters
import spyks.core as spkc


def univariate_noise_dynamical(cf, random_seed=None):

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
    for i in range(cf.data.trials):
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


# if __name__ == "__main__":

#     from munch import Munch
#     import argparse

#     p = argparse.ArgumentParser(description="simulate data from univariate dynamical cascade model")
#     p.add_argument("config", help="path to configuration yaml file")
#     p.add_argument("outfile", help="path to output file for data assimilation")

#     args = p.parse_args()

#     with open(args.config, "rt") as fp:
#         cf = Munch.fromYAML(fp)

#     model_name = os.path.splitext(os.path.basename(cf.data.dynamics.model))[0]
#     pymodel = spkc.load_model(cf.data.dynamics.model)
#     biocm_params = spkc.to_array(pymodel["parameters"])

#     model_dt = cf.model.dt
#     stim_dt = cf.data.dt
#     ntau = cf.model.filter.len

#     k1, kt = filters.gammadiff(ntau * stim_dt / cf.data.filter.taus[0],
#                                ntau * stim_dt / cf.data.filter.taus[1],
#                                cf.data.filter.amplitude,
#                                ntau * stim_dt, stim_dt)

#     # generate data to fit
#     np.random.seed(cf.data.random_seed)
#     mat.random_seed(cf.data.random_seed)
#     assim_data = simulate(cf, k1)
#     test_data = simulate(cf, k1)

#     np.savez(args.outfile,
#              kernel=k1,
#              dynamical_model=model_name,
#              dynamical_params=biocm_params,
#              stim=assim_data[0]["stim"],
#              spikes=np.stack([d["spike_v"] for d in assim_data], axis=1),
#              adapt=np.stack([d["H"] for d in assim_data], axis=2),
#              test_stim=test_data[0]["stim"],
#              test_spikes=np.stack([d["spike_v"] for d in test_data], axis=1),
#              test_adapt=np.stack([d["H"] for d in test_data], axis=2))

#     print("Simulated {} trials in response to white noise. Saved to {}".format(cf.data.trials, args.outfile))
