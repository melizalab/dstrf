# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""This script analyzes the distortions of the RF estimates """
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
import re
from scipy.optimize import differential_evolution
from munch import Munch
from operator import xor

from dstrf import io, strf, mle, simulate, stimulus, filters, models, spikes, performance, crossvalidate, RFfit
import libtfr


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
    p.add_argument("--update-config", "-k",
                   help="set configuration parameter. Use JSON literal. example: -k data.filter.rf=20",
                   action=ParseKeyVal, default=dict(), metavar="KEY=VALUE")   
    p.add_argument("config", help="path to configuration yaml file") 
    p.add_argument("fitfile", help="path to npz file output by assimilate_simulated script")
    p.add_argument("outfile", help="path to output pdf file")


    args = p.parse_args()
    with open(args.config, "rt") as fp:
        cf = Munch.fromYAML(fp) #config file, song_dynamical.yml
    
    for k, v in args.update_config.items():
        path = k.split(".")
        assoc_in(cf, path, v)

    #print("results of assimilation from {}".format(args.fitfile))
    fitfile = np.load(args.fitfile)
    fitfile.allow_pickle=True

    model_info = cf.data.dynamics.model
    model_type = model_info[model_info.find('/') + 1:]
    model_type = model_type[:model_type.find('.yml')]

    rf_type = cf.data.filter.rf

    #Get True Filter
    ncos = cf.model.filter.ncos
    kcosbas = strf.cosbasis(cf.model.filter.len, ncos)
    krank = cf.model.filter.get("rank", None)
    k1, k1t, k1f = simulate.get_filter(cf)

    #Define parameters for FFT
    N = 500
    samp_rate = cf["data"]["dt"]*1000 #Rate of sampling in HZ


    #FFT of Tapered STRF
    def fft_hanning(STRF,samp_rate,N):
        # STRF: Spectrotemporal receptive field to perform FFT on.
        # samp_rate: The temporal sampling rate of your STRF, in Hz.
        # N: How much to upsample or downsample. If not up/downsampling, put in length of STRF temporal profile.
        
        #Signal Description
        nfreq, ntau = STRF.shape
        freq, freq_index = libtfr.fgrid(samp_rate, N)
        
        #Hanning Window of length ntau
        taper = [np.hanning(ntau)]
        
        #Tapered Signals
        tapered_signal = np.zeros_like(STRF)
        for j in np.arange(0,nfreq):
            tapered_signal[j] = STRF[j]*taper
        
        #FFT
        F = np.absolute(np.fft.fft2(tapered_signal,s = (nfreq, N)))
        #Output of Average Spectrum
        output = []
        output.append(freq) # Get relevant frequencies
        output.append(np.sum(F[:,freq_index], 0)) # Sum across frequencies to geth 1D FFT
        output.append(tapered_signal)
        return(output)

    '''
    # Old way
    km, kmt = filters.exponential(46, 1.0, cf.model.filter.len * cf.data.dt, cf.data.dt)
    kconv = np.convolve(km[::-1], k1ft[0,::-1], mode="full")[:km.size][::-1]
    kconv *= k1ft.max() / kconv.max()
    '''

    #FFT of True Time RF
    freq, K1T, tapered_STRF = fft_hanning(k1,samp_rate,N)


    #FFT of Expected Time RF
    km, kmt = filters.exponential(46, 1.0, cf.model.filter.len * cf.data.dt, cf.data.dt)
    kconv = [np.convolve(km[::-1], k1[i,::-1], mode="full")[:km.size][::-1] for i in range(k1.shape[0])]
    kconv = np.row_stack(kconv)

    freq, KCONVT, tapered_STRF_expected = fft_hanning(kconv,samp_rate,N)

    #Get Estimated Filter
    est = fitfile["mle"]
    estparams = est[:3]
    rf_mle = strf.from_basis(strf.defactorize(est[3:], cf.data.filter.nfreq, krank), kcosbas)

    #FFT of Estimated Time RF
    freq, KMLET, tapered_STRF_estimated = fft_hanning(rf_mle,samp_rate,N)

    #Get Peak Distortions
    max_true_idx = K1T.argmax()
    #max_exp_idx = KCONVT.argmax()
    max_est_idx = KMLET.argmax()

    max_true_freq = freq[max_true_idx]
    #max_exp_freq = freq[max_exp_idx]
    max_est_freq = freq[max_est_idx]

    Hz0_true_pwr = K1T[0]/K1T[max_true_idx]
    #Hz0_exp_pwr = KCONVT[0]/KCONVT[max_exp_idx]
    Hz0_est_pwr = KMLET[0]/KMLET[max_est_idx]

    pwr_diff_true = 1-Hz0_true_pwr
    #pwr_diff_exp = 1-Hz0_exp_pwr
    pwr_diff_est = 1-Hz0_est_pwr

    #Get glt amount
    m = re.search(r"glt([0-9]+)_",args.fitfile)

    #Plot Expected vs Estimated
    plt.plot(freq, K1T/K1T.max(), label= "True",color="black")
    #plt.plot(freq, KCONVT/KCONVT.max(), label= "Expected")
    plt.plot(freq, KMLET/KMLET.max(), label = "Estimated",color="red")
    plt.yticks([0,1])
    plt.title("Model: {model}, RF: {rf}".format(model = model_type, rf = rf_type))
    plt.ylabel("Relative Spectral Power")
    plt.xlabel("Frequency (Hz)")
    plt.legend()
    plt.show()
    #plt.savefig('{0}/TimePeakDistortion_{1}_{2}.pdf'.format(args.outfile, model_type, rf_type))

    outrow = [max_true_freq,
              max_est_freq,
              pwr_diff_true,
              pwr_diff_est,
              Hz0_true_pwr,
              Hz0_est_pwr,
              rf_type,
              int(m.group(1))]
   # writer = csv.writer(sys.stdout)
    #writer.writerow(outrow)
    
