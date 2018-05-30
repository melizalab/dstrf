# coding: utf-8

# ## GLMAT: 2D kernel, song stimuli, ML+MC estimation
# 
# This script demonstrates the full assimilation technique using song 
# stimuli. The song waveform is processed to a 2D spectrogram, then convolved 
# with a 2D STRF to produce the "voltage" of the GLMAT model. The adaptation 
# "current" is calculated by convolving the spike trains with two exponential 
# kernels. The goal of the assimilation is to estimate the parameters of the 
# RF and the adaptation kernels. The parameter count of the RF is minimized by 
# using a low-rank approximation (i.e., an outer product of two vectors) and 
# by projecting time into a basis set of raised cosine filters that are spaced 
# exponentially.
# 
# The approach is to use elastic-net penalized maximum-likelihood estimation 
# to get a first guess at the parameters. The regularization parameters and 
# rank are selected using cross-validation. Then MCMC is used to sample the 
# posterior distribution of the parameters.

from __future__ import print_function, division
import os
import sys
import imp
import numpy as np
import progressbar

import yaml
import pickle

import mat_neuron._model as mat
from dstrf import strf, mle, io, performance, spikes
mlfile = sys.argv[1]
saveplace = sys.argv[2]
tag = sys.argv[3]

# The MAT model is governed by a small number of parameters: the spike threshold
# (omega), the amplitudes of the adaptation kernels (alpha_1, alpha_2), the time
# constants of the adaptation kernels (tau_1, tau_2), and the absolute refactory
# period. In addition, a function must be chosen for spike generation. The
# 'softplus' function, log(1 + exp(mu)), is a good choice because it doesn't
# saturate as readily when mu is large. Because there can only be one spike per
# bin, saturation causes the estimated parameters to be less than the true
# parameters.

# We'll use the ML estimate to seed the MCMC sampler. We're going to reduce the
# size of the parameter space by factorizing the RF (i.e., a bilinear
# approximation). Note that we try to use the mlest object as much as possible
# to do the calculations rather than reimplement things; however, there can be
# some significant performance enhancements from an optimized implementation.

try:
  print("Loading ML estimate from {}".format(mlfile))
  with open(mlfile, 'rb') as interfile:
      ml_data = pickle.load(interfile)
except:
  print("Couldn't load ML file {}".format(mlfile))
  sys.exit(1)

yfile = ml_data["yfile"]
cell = ml_data["cell"]
w0 = ml_data["w0"]
assim_data = ml_data["assim_data"]
test_data = ml_data["test_data"]
rf_lambda = ml_data["rf_lambda"]
rf_alpha = ml_data["rf_alpha"]
krank = ml_data["krank"]

with open(yfile,"r") as yf:
    config = yaml.load(yf)
    
# set variables based on `config`
ntaus = len(config["mat"]["taus"])
mat_fixed = np.asanyarray(config["mat"]["taus"] + [config["mat"]["refract"]],dtype='d')
upsample = int(config["strf"]["stim_dt"] / config["mat"]["model_dt"])
kcosbas = strf.cosbasis(config["strf"]["ntau"], config["strf"]["ntbas"])
ntbas = kcosbas.shape[1]


mlest = mle.matfact(assim_data["stim"], kcosbas, krank, assim_data["spike_v"], assim_data["spike_h"],
                        assim_data["stim_dt"], assim_data["spike_dt"], 
                        nlin=config["mat"]["nlin"])

mltest = mle.matfact(test_data["stim"], kcosbas, krank, test_data["spike_v"], test_data["spike_h"],
                     test_data["stim_dt"], test_data["spike_dt"], nlin=config["mat"]["nlin"])

# estimate parameters using emcee
from neurofit import priors, costs, utils, startpos

# the MAT parameters are just bounded between reasonable limits. These may need
# to be expanded when using real data"
mat_prior = priors.joint_independent(
                [ priors.uniform(config["mat"]["bounds"][0][0], config["mat"]["bounds"][0][1]),
                  priors.uniform(config["mat"]["bounds"][1][0], config["mat"]["bounds"][1][1]),
                  priors.uniform(config["mat"]["bounds"][2][0], config["mat"]["bounds"][2][1]),
                ])

def lnpost(theta):
    """Posterior probability for dynamical parameters"""
    mparams = theta[:3]
    rfparams = theta[3:]
    ll = mat_prior(mparams)
    if not np.isfinite(ll):
        return -np.inf
    w = np.r_[mparams, rfparams]
    ll -= mlest.loglike(w, rf_lambda, rf_alpha)
    return -np.inf if not np.isfinite(ll) else ll

print("lnpost of ML estimate: {}".format(lnpost(w0)))
lnpost(w0)

# This code starts the MCMC sampler. We initialize the walkers (chains) in a
# gaussian around the ML estimate, with standard deviation 2x the absolute value
# of the best guess. The model converges fairly quickly, but then we let it
# sample for a while.

import emcee
if sys.platform == 'darwin':
    config["emcee"]["nthreads"] = 1

# initialize walkers
pos = p0 = startpos.normal_independent(config["emcee"]["nwalkers"], w0, np.abs(w0) * 2)
# initialize the sampler
sampler = emcee.EnsembleSampler(config["emcee"]["nwalkers"], w0.size, lnpost, 
                                threads=config["emcee"]["nthreads"])

# start the sampler
tracker = utils.convergence_tracker(config["emcee"]["nsteps"], int(config["emcee"]["nsteps"]/10.0))
for pos, prob, like in tracker(sampler.sample(pos, iterations=config["emcee"]["nsteps"], storechain=True)): 
    continue

print("\n")
print("emcee:")

print("lnpost of p median: {}".format(np.median(prob)))
print("average acceptance fraction: {}".format(sampler.acceptance_fraction.mean()))
try:
    print("autocorrelation time: {}".format(sampler.acor))
except:
    pass    
w1 = np.median(pos, 0)
rfparams = w1[3:]
rf_map = strf.from_basis(mlest.strf(w0), kcosbas)
print(w1[:3])

n_ppost = 10
mat.random_seed(1)
t_stim = np.linspace(0, test_data["duration"], test_data["stim"].shape[1])
    
samples = np.random.permutation(config["emcee"]["nwalkers"])[:n_ppost]
pred = np.zeros((test_data["spike_v"].shape[0], n_ppost), dtype=test_data["spike_v"].dtype)
for i, idx in enumerate(samples):
    mparams = pos[idx]
    V_mc = mltest.V(mparams)
    pred[:, i] = mltest.predict(mparams, mat_fixed, V_mc)
    spk_t = pred[:, i].nonzero()[0]

pred_psth = spikes.psth(pred, upsample, 1)
test_psth = spikes.psth(test_data["spike_v"], upsample, 1)

psth_corr = np.corrcoef(test_psth, pred_psth)[0, 1]
eo = performance.corrcoef(test_data["spike_v"][:,::2], test_data["spike_v"][:,1::2], upsample, 1)

print("loglike: {:.3f}".format(-mltest.loglike(w1)))
print("CC: {:.3f} / {:.3f} ({:.3f})".format(psth_corr, eo, psth_corr/eo))
print("spike count: data = {}, pred = {}".format(test_data["spike_v"].sum() / config["data"]["n_trials"], pred.sum() / n_ppost))

np.savez(saveplace + cell + "_" + tag + "-emcee",chain=sampler.flatchain,lnprob=sampler.flatlnprobability,w1=w1,w0=w0,corr=psth_corr,eo=eo)

