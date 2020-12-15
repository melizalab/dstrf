# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""This script generates predicted responses using the MLE or full posterior """
from __future__ import print_function, division

import os
import sys
import numpy as np
import scipy as sp
import mat_neuron._model as mat
import progressbar
import argparse
import json
import csv
import pandas as pd
from munch import Munch

from dstrf import io, strf, mle, simulate, filters, models, spikes, performance, crossvalidate, RFfit


# plotting packages
import matplotlib.pyplot as plt # plotting functions
import matplotlib.gridspec as grid
import seaborn as sns           # data visualization package
sns.set_style("ticks")

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
    p.add_argument("fitfile", help="path to npz file output by assimilate_simulated script")
    p.add_argument("predfile", help="path to npz file output by predict_simulated script")
    p.add_argument("outfile", help="path to output pdf file")
    p.add_argument("paramfile", help="path to output parameters file")

    args = p.parse_args()
    with open(args.config, "rt") as fp:
        cf = Munch.fromYAML(fp) #config file, song_dynamical.yml
    
    for k, v in args.update_config.items():
        path = k.split(".")
        assoc_in(cf, path, v)

    print("results of assimilation from {}".format(args.fitfile))
    fitfile = np.load(args.fitfile)
    fitfile.allow_pickle=True
 
    print("loading results of prediction from {}".format(args.predfile))
    predfile = np.load(args.predfile)
    predfile.allow_pickle=True
    
    ncos = cf.model.filter.ncos
    kcosbas = strf.cosbasis(cf.model.filter.len, ncos)
    krank = cf.model.filter.get("rank", None)

    k1, k1t, k1f = simulate.get_filter(cf)

    est = fitfile["mle"]
    #est[:3]=[est[0],0,0] # Only do this if you want alphas to be set to zero
    estparams = est[:3]

    model_info = cf.data.dynamics.model
    model_type = model_info[model_info.find('/') + 1:]
    model_type = model_type[:model_type.find('.yml')]

    rf_type = cf.data.filter.rf

    eo = predfile["tspike_corr"]
    predcc = predfile["pspike_corr"]
    
    
    duration_s= predfile["duration"]/1000
    HZ = predfile["pspike_v"].sum()/(duration_s*predfile["ntrials"])
    

    print("MLE rate and adaptation parameters: ", estparams)

    rf_mle = strf.from_basis(strf.defactorize(est[3:], cf.data.filter.nfreq, krank), kcosbas)

    
    #Plot Inspection Plot
    plt.figure(figsize = (7,7))
    g = grid.GridSpec(5,2,height_ratios=[2,1,1,1,1], wspace = 0.1, hspace = 0.1, top = 0.9)

    
    axes = plt.subplot(g[0,0])
    plt.imshow(k1,extent=(k1t[0], k1t[-1], k1f[0], k1f[-1]),cmap='jet',aspect='auto')
    plt.axis("on")
    plt.xticks([])
    plt.yticks([])
    

    axes = plt.subplot(g[0,1])
    plt.imshow(rf_mle,extent=(k1t[0], k1t[-1], k1f[0], k1f[-1]), cmap='jet', aspect='auto')
    plt.xticks([])
    plt.yticks([])



    axes = plt.subplot(g[1,:])
    plt.imshow(predfile["stim"], 
    			   extent = (0, predfile["duration"], cf.data.stimulus.spectrogram.f_min,
    			   cf.data.stimulus.spectrogram.f_max), cmap = 'jet', origin = 'lower',
    			   aspect = 'auto')
    axes.set_xlim(0,2000)
    plt.xticks([])
    plt.yticks([])
    
    t_stim = np.linspace(0, predfile["duration"], predfile["stim"].shape[1])
    t_spike = np.linspace(0, predfile["duration"], predfile["spike_v"].shape[0])

    Vpred = predfile["Vpred"]
    
    axes = plt.subplot(g[2,:])
    plt.plot(t_spike,predfile["V"],'k-')
    axes.set_xlim(0,2000)
    plt.xticks([])
    plt.yticks([])
    


    n_trials = predfile["ntrials"]

    axes = plt.subplot(g[3,:])
    for i, spk in enumerate(predfile["spike_t"]):
    	plt.vlines(spk * cf.model.dt, i - 0.4 + n_trials, i + 0.4 +n_trials)
    pred = np.zeros_like(predfile["spike_v"])
    
    for j in range(n_trials):
    	pred[:, j] = models.predict_spikes_glm(Vpred, est[:3], cf)
    	spk_t = pred[:,j].nonzero()[0]
    	plt.vlines(spk_t * cf.model.dt, j - 0.4, j + 0.4, color = 'r')
    axes.set_xlim(0,2000)
    plt.xticks([])
    plt.yticks([])

    psth_dt = 5
    upsample = int(psth_dt/cf.model.dt)
    pred_psth = spikes.psth(pred, upsample, 1)
    test_psth = spikes.psth(predfile["spike_v"], upsample, 1)
    t_psth = np.linspace(0, predfile["duration"], test_psth.size)

    #Spearman Correlations for no alpha
    #predcc = sp.stats.spearmanr(test_psth, pred_psth)
    #print(predcc)

    # HZ for omega only
    #HZ = pred.sum()/(duration_s*predfile["ntrials"])
    
    axes = plt.subplot(g[4,:])
    plt.plot(t_psth, test_psth,'k-')
    plt.plot( t_psth, pred_psth,'r-' ,alpha = 0.6)
    axes.set_xlim(0,2000)
    sns.despine(top = True, bottom = True, left = True, right = True)
    plt.xticks([])
    plt.yticks([])
    axes.set_xlabel("Model: "+model_type
        +", RF: "+str(rf_type)
        +"\nHZ: "+str(np.around(HZ,2))
        +"\nEOcc: "+str(np.around(eo,2))
        +", Predcc: "+str(np.around(predcc,2))
        +"\n Params:"+str(estparams),
        fontsize="large",fontweight="bold", size=8)
    #plt.show() #Can comment out for batch


    plt.savefig('{0}/songtwin_{1}_{2}.pdf'.format(args.outfile, model_type, rf_type))

    paramfile = args.paramfile

    paramsrow = [model_type,
                 rf_type,
                 estparams[0],
                 estparams[1],
                 estparams[2],
                 eo,
                 predcc,
                 HZ]

    print(",".join(str(v) for v in paramsrow))
    '''
    # This is for alphas equal zero
    paramsrow = [model_type,
             rf_type,
             predcc[0],
             predcc[1],
             HZ]
    '''

    with open(paramfile, 'a') as file:
       writer = csv.writer(file)
       writer.writerow(paramsrow)



