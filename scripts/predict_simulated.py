# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""This script generates predicted responses using the MLE or full posterior """
from __future__ import print_function, division

import sys
import numpy as np
from munch import Munch
import argparse
import json
import csv

from dstrf import io, stimulus, simulate, models, strf, mle, spikes, performance


def predict_mle(mltest, params, cf):
    nbins, ntrials = mltest.spikes.shape
    pred_spikes = np.zeros((nbins, ntrials), dtype=mltest.spikes.dtype)
    V = mltest.V(params)
    for i in range(ntrials):
        S = models.predict_spikes_glm(V, params[:3], cf) 
        pred_spikes[:, i] = S
    return pred_spikes


def predict_posterior(mltest, samples, cf):
    nbins, ntrials = mltest.spikes.shape
    pred_spikes = np.zeros((nbins, ntrials), dtype=mltest.spikes.dtype)
    indices = np.random.permutation(samples.shape[0])[:ntrials]
    for i, idx in enumerate(indices):
        sample = samples[idx]
        V = mltest.V(sample)
        S = models.predict_spikes_glm(V, sample[:3], cf)
        pred_spikes[:, i] = S
    return pred_spikes

def assoc_in(dct, path, value):
    for x in path:
        prev, dct = dct, dct.setdefault(x, {})
    prev[x] = value

class ParseKeyVal(argparse.Action):

    def __call__(self, parser, namespace, arg, option_string=None):
        kv = getattr(namespace, self.dest)
        if kv is None:
            kv = dict()
        if not arg.count('=') == 1:
            raise ValueError(
                "-k %s argument badly formed; needs key=value" % arg)
        else:
            key, val = arg.split('=')
            try:
                kv[key] = json.loads(val)
            except json.decoder.JSONDecodeError:
                kv[key] = val
        setattr(namespace, self.dest, kv)

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="predict responses to test stimuli")
    p.add_argument("--binsize", "-b", type=float, default=10, help="bin size for PSTH (in ms)")
    p.add_argument("--update-config", "-k",
                   help="set configuration parameter. Use JSON literal. example: -k data.trials=50",
                   action=ParseKeyVal, default=dict(), metavar="KEY=VALUE")
    p.add_argument("config", help="path to configuration yaml file")
    p.add_argument("fitfile", help="path to npz file output by assimilate script")
    p.add_argument("outfile", help="path to output npz file")
    p.add_argument("paramfile", help="path to output for estimated parameters")

    args = p.parse_args()
    with open(args.config, "rt") as fp:
        cf = Munch.fromYAML(fp) #config file, song_dynamical.yml

    for k, v in args.update_config.items():
        path = k.split(".")
        assoc_in(cf, path, v)

    model_dt = cf.model.dt
    ncos = cf.model.filter.ncos
    kcosbas = strf.cosbasis(cf.model.filter.len, ncos)

    print("loading results of assimilation from {}".format(args.fitfile))
    fit = np.load(args.fitfile)
    samples = fit["samples"]

    print("loading/generating stimuli")
    stim_fun = getattr(stimulus, cf.data.stimulus.source) #dstrf_sim function from stimulus.py
    data     = stim_fun(cf) #stimulus.dstrf_sim(cf), this is just the song

    p_test = cf.data.get("test_proportion", None)
    if p_test is None:
        test_data  = stim_fun(cf, random_seed=1000)
    else:
        n_test = int(p_test * len(data))
        print("using last {} stimuli for test".format(n_test))
        test_data = data[-n_test:]

    data_fun = getattr(simulate, cf.data.model) #multivariate_dynamical
    print("simulating response for testing using {}".format(cf.data.model))
    tdata = io.merge_data(data_fun(cf, test_data, random_seed=1000))

    # we use the estimator to generate predictions
    try:
        mltest = mle.mat(tdata["stim"], kcosbas, tdata["spike_v"], tdata["spike_h"], tdata["stim_dt"], tdata["spike_dt"])
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
        pred_spikes = predict_posterior(mltest, samples, cf)
    else:
        print("generating predictions from MLE")
        pred_spikes = predict_mle(mltest, fit["mle"], cf) #fit["mle"] are the true parameters

    est = np.median(samples,axis=0) #Use this to make the Vpred variable

    Vpred = mltest.V(est)
    tspike_v = tdata["spike_v"]
    upsample = int(args.binsize / cf.model.dt)
    test_psth = spikes.psth(tspike_v, upsample, 1)
    pred_psth = spikes.psth(pred_spikes, upsample, 1)

    eo = performance.corrcoef(tspike_v[::2], tspike_v[1::2], upsample, 1)
    cc = np.corrcoef(test_psth, pred_psth)[0, 1]
    print("Prediction performance (dt = {:2} ms)".format(args.binsize))
    print("EO cc: %3.3f" % eo)
    print("pred cc: %3.3f" % cc)
    print("spike count: data = {}, pred = {}".format(tspike_v.sum(), pred_spikes.sum())) #is showing wrong number of pred spikes

    #Get predicted spike times
    spk_t = []
    for j in range(tdata["ntrials"]):
        pred_spikes[:, j] = models.predict_spikes_glm(Vpred, est[:3], cf)
        spk_t.append(pred_spikes[:, j].nonzero()[0])

    #Get sum of spikes per trial
    sum_sim = tspike_v.sum(axis=0)
    sum_pred = pred_spikes.sum(axis=0)

    #Get mean spike sum per trial
    sim_mean = np.mean(sum_sim)
    pred_mean = np.mean(sum_pred)

    #Get coefficient of variation of spike sum per trial
    sim_cv = np.std(sum_sim)/sim_mean
    pred_cv = np.std(sum_pred)/pred_mean 

    isi_pred = np.concatenate([np.diff(spikes) for spikes in spk_t])
    isi_sim  = np.concatenate([np.diff(spikes) for spikes in tdata["spike_t"]])
    #Get ISI for each trial
    isi_pred = []
    for j in range(tdata["ntrials"]):
        for i in range(len(spk_t[j])-1):
            isi_pred.append(abs(spk_t[j][i]-spk_t[j][i+1]))
    isi_sim = []
    for j in range(tdata["ntrials"]):
        for i in range(len(tdata["spike_t"][j])-1):
            isi_sim.append(abs(tdata["spike_t"][j][i]-tdata["spike_t"][j][i+1]))

    #Get mean isi across trials
    isi_pred_mean = np.mean(isi_pred)
    isi_sim_mean = np.mean(isi_sim)

    #Get coefficient of variation of ISI across trials
    isi_pred_cv = np.std(isi_pred)/isi_pred_mean
    isi_sim_cv = np.std(isi_sim)/isi_sim_mean
                
    out = {"spike_v":tspike_v, "duration":tdata["duration"], 
           "spike_h":tdata["spike_h"],"stim_dt":tdata["stim_dt"],
           "spike_dt":tdata["spike_dt"],"stim":tdata["stim"], 
           "V":tdata["V"], "ntrials":tdata["ntrials"], 
           "spike_t":tdata["spike_t"],
           "pred spikes":pred_spikes, 
           "sim psth":test_psth, "pred psth":pred_psth,
           "estimates":est,
           "Vpred":Vpred,
           "eo cc":eo,
           "pred cc":cc}
    np.savez(args.outfile, **out)

    duration_s= tdata["duration"]/1000
    HZ = pred_spikes.sum()/(duration_s*tdata["ntrials"])

    estparams = est

    rf_type = cf.data.filter.rf

    model_info = cf.data.dynamics.model
    model_type = model_info[model_info.find('/') + 1:]
    model_type = model_type[:model_type.find('.yml')]

    paramfile = args.paramfile
    paramsrow = [model_type,
                 rf_type,
                 estparams[0],
                 estparams[1],
                 estparams[2],
                 eo,
                 cc,
                 HZ,
                 sim_mean,
                 pred_mean,
                 sim_cv,
                 pred_cv,
                 isi_sim_mean,
                 isi_pred_mean,
                 isi_sim_cv,
                 isi_pred_cv]
    with open(paramfile, 'a') as file:
       writer = csv.writer(file)
       writer.writerow(paramsrow)
  
