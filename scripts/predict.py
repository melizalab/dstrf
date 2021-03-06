# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""This script generates predicted responses using the MLE or full posterior """
from __future__ import print_function, division

import numpy as np
from munch import Munch

from dstrf import io, data, simulate, models, strf, mle, spikes, performance, util


def predict_mle(mltest, params, cf):
    nbins, ntrials = mltest.spikes.shape
    try:
        ntrials = cf.data.test.trials
    except AttributeError:
        pass
    pred_spikes = np.zeros((nbins, ntrials), dtype=mltest.spikes.dtype)
    V = mltest.V(params)
    for i in range(ntrials):
        S = models.predict_spikes_glm(V, params[:3], cf)
        pred_spikes[:, i] = S
    return pred_spikes


def predict_posterior(mltest, samples, cf):
    nbins, ntrials = mltest.spikes.shape
    try:
        ntrials = cf.data.test.trials
    except AttributeError:
        pass
    pred_spikes = np.zeros((nbins, ntrials), dtype=mltest.spikes.dtype)
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
    p.add_argument("--binsize", "-b", type=float, default=10, help="bin size for PSTH (in ms)")
    p.add_argument("--params", "-p",
                   help="output spike history parameters and performance data to this file in json format")
    p.add_argument("--save-data", help="save the test data and prediction to this file")
    p.add_argument("--update-config", "-k",
                   help="set configuration parameter. Use JSON literal. example: -k data.filter.rf=20",
                   action=util.ParseKeyVal, default=dict(), metavar="KEY=VALUE")
    p.add_argument("config", help="path to configuration yaml file")
    p.add_argument("fitfile", help="path to npz file output by assimilate script")

    args = p.parse_args()
    with open(args.config, "rt") as fp:
        cf = Munch.fromYAML(fp)

    for k, v in args.update_config.items():
        path = k.split(".")
        util.assoc_in(cf, path, v)

    model_dt = cf.model.dt
    ncos = cf.model.filter.ncos
    kcosbas = strf.cosbasis(cf.model.filter.len, ncos)

    print("loading results of assimilation from {}".format(args.fitfile))
    fit = np.load(args.fitfile)

    print("loading/generating data using", cf.data.source)
    seed = cf.data.stimulus.random_seed = 19924
    print(" - setting random seed for test stimulus to {}".format(seed))
    stim_fun = getattr(data, cf.data.source)
    data     = stim_fun(cf)

    if "model" in cf.data:
        print("simulating response for testing using {}".format(cf.data.model))
        data_fun = getattr(simulate, cf.data.model)
        test_data = data_fun(cf, data,
                            random_seed=cf.data.test.random_seed,
                            trials=cf.data.test.trials)

    try:
        p_test = cf.data.test.proportion
    except AttributeError:
        p_test = None
    test_data = io.subselect_data(data, p_test, first=False)
    tdata = io.merge_data(test_data)

    print(" - duration:", tdata["duration"])
    print(" - stim bins:", tdata["stim"].shape[1])
    print(" - trials:", tdata["spike_v"].shape[1])
    print(" - spike bins:", tdata["spike_v"].shape[0])
    print(" - total spikes:", np.sum(tdata["spike_v"]))

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

    matboundprior = models.matbounds(cf.model.ataus[0], cf.model.ataus[1], cf.model.t_refract)
    if "samples" in fit:
        print("generating posterior predictive distribution")
        pred_spikes = predict_posterior(mltest, fit["samples"], cf)
        params = np.median(fit["samples"], 0)
        allowed = [matboundprior(s) for s in fit["samples"]]
        in_bounds = np.mean(allowed)
    else:
        print("generating predictions from MLE")
        print(" - MLE rate and adaptation parameters:", fit["mle"][:3])
        params = fit["mle"]
        in_bounds = matboundprior(params)
        if not in_bounds:
            print(" - warning: parameter estimates not in allowed region")
        pred_spikes = predict_mle(mltest, fit["mle"], cf)

    tspike_v = tdata["spike_v"]
    upsample = int(args.binsize / cf.model.dt)
    test_psth = spikes.psth(tspike_v, upsample, 1)
    pred_psth = spikes.psth(pred_spikes, upsample, 1)
    duration_s= tdata["duration"]/1000
    HZ = pred_spikes.sum()/(duration_s*tdata["ntrials"])

    Vpred = mltest.V(fit["mle"])

    eo = performance.corrcoef(tspike_v[::2], tspike_v[1::2], upsample, 1)
    cc = np.corrcoef(test_psth, pred_psth)[0, 1]
    print("Prediction performance (dt = {:2} ms)".format(args.binsize))
    print(" - EO cc: %3.3f" % eo)
    print(" - pred cc: %3.3f" % cc)
    print(" - spike count: data = {}, pred = {} ({} Hz)".format(tspike_v.sum(0).mean(), pred_spikes.sum(0).mean(), HZ))
    print(" - omega =", params[0])
    print(" - a1 =", params[1])
    print(" - a2 =", params[2])
    print(" - mean data rate =", tspike_v.sum(0).mean())
    print(" - mean pred rate =", pred_spikes.sum(0).mean())
    print(" - rate_sd_data =", tspike_v.sum(0).std())
    print(" - rate_sd_pred =", pred_spikes.sum(0).std())

    if args.save_data:
        print("Saving prediction data to", args.save_data)
        tdata["tspike_psth"] = test_psth
        tdata["pspike_v"] = pred_spikes
        tdata["pspike_psth"] = pred_psth
        tdata["tspike_corr"] = eo
        tdata["pspike_corr"] = cc
        tdata["Vpred"] = Vpred
        np.savez(args.save_data, **tdata)

    if args.params:
        import json
        output = {
            "w": params[0],
            "a1": params[1],
            "a2": params[2],
            "duration": tdata["duration"],
            "binsize": args.binsize,
            "cor_data": eo,
            "cor_pred": cc,
            "spike_mean_data": tspike_v.sum(0).mean(),
            "spike_mean_pred": pred_spikes.sum(0).mean(),
            "spike_sd_data": tspike_v.sum(0).std(),
            "spike_sd_pred": pred_spikes.sum(0).std(),
            "trials_data": tspike_v.shape[1],
            "trials_pred": pred_spikes.shape[1],
            "params_in_bounds": in_bounds * 1.0
        }
        with open(args.params, "wt") as fp:
            json.dump(output, fp)
