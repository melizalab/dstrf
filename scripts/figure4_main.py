# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""This script generates plots for figure 4 """
from __future__ import print_function, division

import os
import sys
import numpy as np
import mat_neuron._model as mat
import progressbar
import argparse
import json
import csv
import pandas as pd
from scipy.optimize import differential_evolution
from munch import Munch

from dstrf import io, strf, mle, simulate, filters, models, spikes, performance, crossvalidate, RFfit


# plotting packages
import matplotlib.pyplot as plt # plotting functions
import matplotlib.gridspec as grid
import seaborn as sns           # data visualization package
sns.set_style("ticks")
est_clr = ["darkred","darkmagenta","chocolate"]

sns.set_context("paper", font_scale=0.7)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 1.5
plt.rcParams['ytick.major.size'] = 1.5

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

    import argparse

    p = argparse.ArgumentParser(description="plot predicted and simulated spikes and STRFs")
    p.add_argument("config", help="path to configuration yaml file")
    p.add_argument("--update-config", "-k",
                   help="set configuration parameter. Use JSON literal. example: -k data.filter.rf=20",
                   action=ParseKeyVal, default=dict(), metavar="KEY=VALUE") 
    p.add_argument("danfile", help="path to npz file output from dan")
    p.add_argument("predfile", help="path to npz file output by predict_simulated script")
    p.add_argument("outfile", help="path to output pdf file")

    args = p.parse_args()
    with open(args.config, "rt") as fp:
        cf = Munch.fromYAML(fp) #config file, song_dynamical.yml
    
    for k, v in args.update_config.items():
        path = k.split(".")
        assoc_in(cf, path, v)

    #print("results of assimilation from {}".format(args.fitfile))
    danfile = np.load(args.danfile)
    #fitfile.allow_pickle=True
 
    print("loading results of prediction from {}".format(args.predfile))
    predfile = np.load(args.predfile)
    predfile.allow_pickle=True
    
    ncos = cf.model.filter.ncos
    kcosbas = strf.cosbasis(cf.model.filter.len, ncos)
    krank = cf.model.filter.get("rank", None)

    k1, k1t, k1f = simulate.get_filter(cf)

    est = predfile["mle"]
    estparams = est[:3]

    model_info = cf.data.dynamics.model
    model_type = model_info[model_info.find('/') + 1:]
    model_type = model_type[:model_type.find('.yml')]

    rf_type = cf.data.filter.rf
    rf_mle = strf.from_basis(strf.defactorize(est[3:], cf.data.filter.nfreq, krank), kcosbas)

    if(rf_type == 4):
        est_clr = "darkred"
    elif(rf_type ==22):
        est_clr = "darkmagenta"
    else:
        est_clr = "chocolate"
    plt.figure(figsize = (7,7))
    g = grid.GridSpec(3,2,height_ratios=[1,1,1], wspace = 0.1, hspace = 0.1, top = 0.9)

    
    axes = plt.subplot(g[0,0])
    plt.imshow(k1,extent=(k1t[0], k1t[-1], k1f[0], k1f[-1]),cmap='jet',aspect='auto')
    plt.xticks([])
    plt.yticks([])
    

    axes = plt.subplot(g[0,1])
    plt.imshow(rf_mle,extent=(k1t[0], k1t[-1], k1f[0], k1f[-1]), cmap='jet', aspect='auto')
    plt.xticks([])
    plt.yticks([])

    #Dan file stuff
    data = danfile
    V = data["V"]
    stim = data["stim"].squeeze()
    tspk = data["spike_v"]
    pspk = data["pspike_v"]
    ntrials = min(tspk.shape[1], 10)

    upsample = int(cf.data.dt / cf.model.dt)
    test_psth = spikes.psth(tspk, upsample, 1)
    pred_psth = spikes.psth(pspk, upsample, 1)
    t_psth = np.linspace(0, data["duration"], test_psth.size)

    #Raster
    axes = plt.subplot(g[1,:])
    for i in range(ntrials):
        spk_t = np.nonzero(tspk[:, i])[0] * cf.model.dt
        plt.vlines(spk_t, i - 0.4 + ntrials, i + 0.4 + ntrials)
    for i in range(ntrials):
        spk_t = np.nonzero(pspk[:, i])[0] * cf.model.dt
        plt.vlines(spk_t, i - 0.4, i + 0.4, color=est_clr)

    axes.set_xlim(5950, 8050);


    # PSTHs
    axes = plt.subplot(g[2,:])
    plt.plot(t_psth, test_psth, linewidth=1, color='k', label="data")
    plt.plot(t_psth, pred_psth, linewidth=1, color=est_clr, label="data")
    axes.set_xlim(5950,8050)
    plt.show() #Can comment out for batch


    #plt.savefig('{0}/songtwin_{1}_{2}.pdf'.format(args.outfile, model_type, rf_type))

