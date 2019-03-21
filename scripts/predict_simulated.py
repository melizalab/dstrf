# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""This script generates predicted responses using the MLE or full posterior """
from __future__ import print_function, division

import sys
import numpy as np
from munch import Munch

from dstrf import io, stimulus, simulate, models, strf, mle, spikes, performance


def predict_mle(mltest, params, cf):
    nbins, ntrials = mltest.spikes.shape
    pred_spikes = np.zeros((nbins, ntrials), dtype=mlest.spikes.dtype)
    V = mltest.V(params)
    for i in range(ntrials):
        S = models.predict_spikes_glm(V, params[:3], cf)
        pred_spikes[:, i] = S
    return pred_spikes


def predict_posterior(mltest, samples, cf):
    nbins, ntrials = mltest.spikes.shape
    pred_spikes = np.zeros((nbins, ntrials), dtype=mlest.spikes.dtype)
    indices = np.random.permutation(cf.emcee.nwalkers)[:ntrials]
    for i, idx in enumerate(indices):
        sample = samples[idx]
        V = mltest.V(sample)
        S = models.predict_spikes_glm(V, sample[:3], cf)
        pred_spikes[:, i] = S
    return pred_spikes


if __name__ == "__main__":

    import argparse

    p = argparse.ArgumentParser(description="predict responses to test stimuli")
    p.add_argument("--binsize", "-b", type=float, dfeault=10, help="bin size for PSTH (in ms)")
    p.add_argument("config", help="path to configuration yaml file")
    p.add_argument("fitfile", help="path to npz file output by assimilate script")
    p.add_argument("outfile", help="path to output npz file")

    args = p.parse_args()
    with open(args.config, "rt") as fp:
        cf = Munch.fromYAML(fp)

    model_dt = cf.model.dt
    ncos = cf.model.filter.ncos
    kcosbas = strf.cosbasis(cf.model.filter.len, ncos)

    print("loading results of assimilation from {}".format(args.fitfile))
    fit = np.load(args.fitfile)

    print("loading/generating stimuli")
    stim_fun = getattr(stimulus, cf.data.stimulus.source)
    data     = stim_fun(cf)

    p_test = cf.data.get("test_proportion", None)
    if p_test is None:
        test_data  = stim_fun(cf, random_seed=1000)
    else:
        n_test = int(p_test * len(data))
        print("using last {} stimuli for test".format(n_test))
        test_data = data[-n_test:]

    data_fun = getattr(simulate, cf.data.model)
    print("simulating response for testing using {}".format(cf.data.model))
    tdata = io.merge_data(data_fun(cf, test_data, random_seed=1000))

    # we use the estimator to generate predictions
    try:
        mlest = mle.mat(tdata["stim"], kcosbas, tdata["spike_v"], tdata["spike_h"], tdata["stim_dt"], tdata["spike_dt"])
    except TypeError:
        pass
    krank = cf.model.filter.get("rank", None)
    if krank is None:
        mltest = mle.mat(tdata["stim"], kcosbas, tdata["spike_v"], tdata["spike_h"],
                         tdata["stim_dt"], tdata["spike_dt"])
    else:
        mltest = mle.matfact(tdata["stim"], kcosbas, krank, tdata["spike_v"], tdata["spike_h"],
                             tdata["stim_dt"], tdata["spike_dt"])

    if "samples" in fit:
        print("generating posterior predictive distribution")
        pred_spikes = predict_posterior(mltest, fit["samples"], cf)
    else:
        print("generating predictions from MLE")
        pred_spikes = predict_mle(mltest, fit["mle"], cf)

    tspike_v = tdata["spike_v"]
    upsample = int(args.binsize / cf.model.dt)
    test_psth = spikes.psth(tspike_v, upsample, 1)
    pred_psth = spikes.psth(pred_spikes, upsample, 1)

    eo = performance.corrcoef(tspike_v[::2], tspike_v[1::2], upsample, 1)
    cc = np.corrcoef(test_psth, pred_psth)[0, 1]
    print("Prediction performance (dt = {:2} ms)".format(args.binsize))
    print("EO cc: %3.3f" % eo)
    print("pred cc: %3.3f" % cc)
    print("spike count: data = {}, pred = {}".format(tspike_v.sum(), pred_spikes.sum()))
