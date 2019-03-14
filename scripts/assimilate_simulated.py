# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" This script will do a full emcee estimate from simulated data """
from __future__ import print_function, division

import sys
import numpy as np
from munch import Munch
import emcee

from neurofit import priors, utils, startpos
from dstrf import io, stimulus, simulate, models, strf, mle, spikes, performance


if __name__ == "__main__":

    import argparse

    p = argparse.ArgumentParser(description="sample from posterior of simulated dat")
    p.add_argument("config", help="path to configuration yaml file")
    p.add_argument("outfile", help="path to output npz file")

    args = p.parse_args()

    with open(args.config, "rt") as fp:
        cf = Munch.fromYAML(fp)

    model_dt = cf.model.dt
    ncos = cf.model.filter.ncos
    kcosbas = strf.cosbasis(cf.model.filter.len, ncos)

    print("loading/generating stimuli")
    stim_fun = getattr(stimulus, cf.data.stimulus.source)
    data     = stim_fun(cf)

    print("simulating response using {}".format(cf.data.model))
    data_fun = getattr(simulate, cf.data.model)
    data = io.merge_data(data_fun(cf, data))
    print("spike count: {}".format(data["spike_v"].sum()))

    # this always fails on the first try for reasons I don't understand
    try:
        mlest = mle.mat(data["stim"], kcosbas, data["spike_v"], data["spike_h"], data["stim_dt"], data["spike_dt"])
    except TypeError:
        pass
    krank = cf.model.filter.get("rank", None)
    if krank is None:
        print("starting ML estimation - full rank")
        mlest = mle.mat(data["stim"], kcosbas, data["spike_v"], data["spike_h"], data["stim_dt"], data["spike_dt"])
    else:
        print("starting ML estimation - rank={}".format(krank))
        mlest = mle.matfact(data["stim"], kcosbas, krank, data["spike_v"], data["spike_h"], data["stim_dt"], data["spike_dt"])

    w0 = mlest.estimate(reg_alpha=cf.regularization.alpha, reg_lambda=cf.regularization.lmbda)
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
    rf_lambda = cf.regularization.lmbda
    rf_alpha = cf.regularization.alpha

    def lnpost(theta):
        """Posterior probability"""
        mparams = theta[:3]
        if not matboundprior(mparams):
            return -np.inf
        lp = mat_prior(mparams)
        if not np.isfinite(lp):
            return -np.inf
        # mlest can do penalty for lambda
        ll = mlest.loglike(theta, rf_lambda, rf_alpha)
        if not np.isfinite(ll):
            return -np.inf
        return lp - ll

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
    print("loading/generating stimuli for validation")
    tdata     = stim_fun(cf, random_seed=1000)

    print("simulating response using {}".format(cf.data.model))
    tdata = io.merge_data(data_fun(cf, tdata, random_seed=1000))

    # we use the estimator to generate predictions
    if krank is None:
        mltest = mle.mat(tdata["stim"], kcosbas, tdata["spike_v"], tdata["spike_h"], tdata["stim_dt"], tdata["spike_dt"])
    else:
        mltest = mle.matfact(tdata["stim"], kcosbas, krank, tdata["spike_v"], tdata["spike_h"], tdata["stim_dt"], tdata["spike_dt"])

    tspike_v = tdata["spike_v"]
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

    eo = performance.corrcoef(tspike_v[::2], tspike_v[1::2], upsample, 1)
    cc = np.corrcoef(test_psth, pred_psth)[0, 1]
    print("EO cc: %3.3f" % eo)
    print("pred cc: %3.3f" % cc)
    print("spike count: data = {}, pred = {}".format(tspike_v.sum(), pred_spikes.sum()))

    np.savez(args.outfile,
             astim=data["stim"], aspikes=data["spike_v"],
             pos=pos, prob=prob, eo=eo, cc=cc,
             tstim=tdata["stim"], tspikes=tdata["spike_v"],
             pspikes=np.column_stack(pred_spikes))
