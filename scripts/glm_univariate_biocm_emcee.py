# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" This script will do a full emcee estimate from the univariate biocm model """
from __future__ import print_function, division

import sys
import os
import numpy as np
import quickspikes as qs
from munch import Munch
import emcee

import mat_neuron._model as mat
from neurofit import priors, costs, utils, startpos
from dstrf import strf, mle, filters, spikes, performance
import spyks.core as spkc


def simulate(cf, kernel):
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


def matbounds(t1, t2, tr):
    """Returns function for boundaries on adaptation parameters based on disallowed
    region (see Yamauchi et al 2011)

    """
    aa1 = -(1 - np.exp(-tr / t2)) / (1 - np.exp(-tr / t1))
    aa2 = -(np.exp(tr / t2) - 1) / (np.exp(tr / t1) - 1)
    return lambda mparams: (mparams[2] > aa1 * mparams[1]) and (mparams[2] > aa2 * mparams[1])


def predict_spikes_glm(V, params, cf):
    """Predict spikes using GLM from voltage and rate/adaptation params"""
    upsample = int(cf.data.dt / cf.model.dt)
    omega, a1, a2  = params
    return mat.predict_poisson(V - omega, (a1, a2),
                               cf.model.ataus, cf.model.t_refract, cf.model.dt, upsample)


if __name__ == "__main__":

    import argparse

    p = argparse.ArgumentParser(description="sample from posterior of glm_univariate_biocm")
    p.add_argument("config", help="path to configuration yaml file")

    args = p.parse_args()

    with open(args.config, "rt") as fp:
        cf = Munch.fromYAML(fp)

    model_name = os.path.splitext(os.path.basename(cf.data.dynamics.model))[0]

    model_dt = cf.model.dt
    stim_dt = cf.data.dt
    ntau = cf.model.filter.len
    ncos = cf.model.filter.ncos
    kcosbas = strf.cosbasis(ntau, ncos)

    k1, kt = filters.gammadiff(ntau * stim_dt / cf.data.filter.taus[0],
                               ntau * stim_dt / cf.data.filter.taus[1],
                               cf.data.filter.amplitude,
                               ntau * stim_dt, stim_dt)

    # data parameters

    # this needs to be adjusted on a per model basis. posp ~ 2.0; phasic ~ 10
    # model_name = "biocm_phasic"
    # current_scaling = 9.0
    # model_name = "biocm_tonic"
    # current_scaling = 4.0
    # model_name = "pospischil_sm"
    # current_scaling = 2.0

    # generate data to fit
    np.random.seed(1)
    mat.random_seed(1)
    assim_data = simulate(cf, k1)


    # initial guess of parameters using regularized ML
    stim = assim_data[0]["stim"]
    spike_v = np.stack([d["spike_v"] for d in assim_data], axis=1)
    spike_h = np.stack([d["H"] for d in assim_data], axis=2)
    try:
        mlest = mle.mat(stim, kcosbas, spike_v, spike_h, stim_dt, model_dt, nlin="exp")
    except TypeError:
        mlest = mle.mat(stim, kcosbas, spike_v, spike_h, stim_dt, model_dt, nlin="exp")
    w0 = mlest.estimate(reg_alpha=1.0)
    print("MLE rate and adaptation parameters:", w0[:3])

    # estimate parameters using emcee
    if sys.platform == 'darwin':
        cf.emcee.nthreads = 1

    # set up priors - base rate and adaptation
    mat_prior = priors.joint_independent([priors.uniform(0, 20),
                                          priors.uniform(-50, 200),
                                          priors.uniform(-5, 10)])
    # additional constraint to stay out of disallowed regio
    matboundprior = matbounds(cf.model.ataus[0], cf.model.ataus[1], cf.model.t_refract)

    # lasso prior on RF parameters
    rf_lambda = 1.0

    def lnprior(theta):
        mparams = theta[:3]
        rfparams = theta[3:]
        if not matboundprior(mparams):
            return -np.inf
        rf_prior = -np.sum(np.abs(rfparams)) * rf_lambda
        ll = mat_prior(mparams) + rf_prior
        if not np.isfinite(ll):
            return -np.inf
        else:
            return ll

    def lnpost_dyn(theta):
        """Posterior probability for dynamical parameters"""
        return lnprior(theta) - mlest.loglike(theta)

    # initial state is a gaussian ball around the ML estimate
    p0 = startpos.normal_independent(cf.emcee.nwalkers, w0, np.abs(w0) * cf.emcee.startpos_scale)
    theta_0 = np.median(p0, 0)
    print("lnpost of p0 median: {}".format(lnpost_dyn(theta_0)))

    sampler = emcee.EnsembleSampler(cf.emcee.nwalkers, w0.size, lnpost_dyn,
                                    threads=cf.emcee.nthreads)
    tracker = utils.convergence_tracker(cf.emcee.nsteps, 25)

    for pos, prob, _ in tracker(sampler.sample(p0, iterations=cf.emcee.nsteps)):
        continue

    print("lnpost of p median: {}".format(np.median(prob)))
    print("average acceptance fraction: {}".format(sampler.acceptance_fraction.mean()))
    theta = np.median(pos, 0)
    print("MAP rate and adaptation parameters:", theta[:3])

    # simulate test data and posterior predictions
    np.random.seed(1000)
    mat.random_seed(1000)
    test_data = simulate(cf, k1)

    tstim = test_data[0]["stim"]
    tspike_v = np.stack([d["spike_v"] for d in test_data], axis=1)
    tspike_h = np.stack([d["H"] for d in test_data], axis=2)
    mltest = mle.mat(tstim, kcosbas, tspike_v, tspike_h, stim_dt, model_dt, nlin="exp")

    pred_spikes = np.zeros_like(tspike_v)
    samples = np.random.permutation(cf.emcee.nwalkers)[:cf.data.trials]
    for i, idx in enumerate(samples):
        sample = pos[idx]
        V = mltest.V(sample)
        S = predict_spikes_glm(V, sample[:3], cf)
        pred_spikes[:, i] = S

    upsample = int(cf.data.dt / cf.model.dt)
    test_psth = spikes.psth(tspike_v, upsample, 1)
    pred_psth = spikes.psth(pred_spikes, upsample, 1)

    eo = performance.corrcoef(spike_v[::2], spike_v[1::2], upsample, 1)
    cc = np.corrcoef(test_psth, pred_psth)[0, 1]
    print("CC: {}/{} = {}".format(cc, eo, cc / eo))

    outfile = os.path.join("results", "{}_samples.npz".format(model_name))
    np.savez(outfile,
             astim=assim_data[0]["stim"], acurrent=assim_data[0]["I"], astate=assim_data[0]["state"], aspikes=spike_v,
             pos=pos, prob=prob, eo=eo, cc=cc,
             tstim=tstim, tcurrent=test_data[0]["I"], tstate=test_data[0]["state"], tspikes=tspike_v,
             pspikes=np.column_stack(pred_spikes))
