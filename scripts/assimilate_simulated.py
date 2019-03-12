# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" This script will do a full emcee estimate from simulated data """
from __future__ import print_function, division

import sys
import os
import numpy as np
from munch import Munch
import emcee

from neurofit import priors, utils, startpos
from dstrf import simulate, models, strf, mle, spikes, performance


if __name__ == "__main__":

    import argparse

    p = argparse.ArgumentParser(description="sample from posterior of simulated dat")
    p.add_argument("config", help="path to configuration yaml file")
    p.add_argument("outfile", help="path to output npz file")

    args = p.parse_args()

    with open(args.config, "rt") as fp:
        cf = Munch.fromYAML(fp)

    try:
        data_fun = getattr(simulate, cf.data.source)
    except AttributeError:
        print("no function called {} in the simulation module".format(cf.data.source))
        sys.exit(-1)

    model_name = os.path.splitext(os.path.basename(cf.data.dynamics.model))[0]

    model_dt = cf.model.dt
    stim_dt = cf.data.dt
    ncos = cf.model.filter.ncos
    kcosbas = strf.cosbasis(cf.model.filter.len, ncos)

    print("simulating data from {} function".format(cf.data.source))
    assim_data = data_fun(cf)
    stim = assim_data[0]["stim"]
    spike_v = np.stack([d["spike_v"] for d in assim_data], axis=1)
    spike_h = np.stack([d["H"] for d in assim_data], axis=2)

    print("starting ML estimation")
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
    matboundprior = models.matbounds(cf.model.ataus[0], cf.model.ataus[1], cf.model.t_refract)

    # lasso prior on RF parameters
    rf_lambda = 1.0

    def lnpost(theta):
        """Posterior probability"""
        mparams = theta[:3]
        if not matboundprior(mparams):
            return -np.inf
        lp = mat_prior(mparams)
        if not np.isfinite(lp):
            return -np.inf
        # mlest can do penalty for lambda
        return lp - mlest.loglike(theta, rf_lambda)

    # initial state is a gaussian ball around the ML estimate
    p0 = startpos.normal_independent(cf.emcee.nwalkers, w0, np.abs(w0) * cf.emcee.startpos_scale)
    theta_0 = np.median(p0, 0)
    print("lnpost of p0 median: {}".format(lnpost(theta_0)))

    sampler = emcee.EnsembleSampler(cf.emcee.nwalkers, w0.size, lnpost,
                                    threads=cf.emcee.nthreads)
    tracker = utils.convergence_tracker(cf.emcee.nsteps, 25)

    for pos, prob, _ in tracker(sampler.sample(p0, iterations=cf.emcee.nsteps)):
        continue

    print("lnpost of p median: {}".format(np.median(prob)))
    print("average acceptance fraction: {}".format(sampler.acceptance_fraction.mean()))
    theta = np.median(pos, 0)
    print("MAP rate and adaptation parameters:", theta[:3])

    # simulate test data and posterior predictions
    print("simulating data for validation from {} function".format(cf.data.source))
    test_data = data_fun(cf, random_seed=1000)
    tstim = test_data[0]["stim"]
    tspike_v = np.stack([d["spike_v"] for d in test_data], axis=1)
    tspike_h = np.stack([d["H"] for d in test_data], axis=2)

    mltest = mle.mat(tstim, kcosbas, tspike_v, tspike_h, stim_dt, model_dt, nlin="exp")
    pred_spikes = np.zeros_like(tspike_v)
    samples = np.random.permutation(cf.emcee.nwalkers)[:cf.data.trials]
    for i, idx in enumerate(samples):
        sample = pos[idx]
        V = mltest.V(sample)
        S = models.predict_spikes_glm(V, sample[:3], cf)
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
