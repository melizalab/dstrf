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

import yaml

import mat_neuron._model as mat
from dstrf import strf, mle, io, performance, spikes
cell = sys.argv[1]
yfile = sys.argv[2]
saveplace = sys.argv[3]
tag = sys.argv[4]

# The MAT model is governed by a small number of parameters: the spike threshold
# (omega), the amplitudes of the adaptation kernels (alpha_1, alpha_2), the time
# constants of the adaptation kernels (tau_1, tau_2), and the absolute refactory
# period. In addition, a function must be chosen for spike generation. The
# 'softplus' function, log(1 + exp(mu)), is a good choice because it doesn't
# saturate as readily when mu is large. Because there can only be one spike per
# bin, saturation causes the estimated parameters to be less than the true
# parameters.

with open(yfile,"r") as yf:
    config = yaml.load(yf)
    
# set variables based on `config`
ntaus = len(config["mat"]["taus"])
mat_fixed = np.asanyarray(config["mat"]["taus"] + [config["mat"]["refract"]],dtype='d')
upsample = int(config["strf"]["stim_dt"] / config["mat"]["model_dt"])
kcosbas = strf.cosbasis(config["strf"]["ntau"], config["strf"]["ntbas"])
ntbas = kcosbas.shape[1]

# Here we load some data from a real neural recording from the CRCNS dataset. To
# simplify the model, we concatenate the stimuli, setting padding between the
# stimuli sufficient to capture any offset responses. Note that the spike
# responses are convolved with the adaptation kernels before merging stimuli so
# that we don't inadvertently carry over spike history from trials that are not
# truly contiguous.

pad_after = config["strf"]["ntau"] * config["strf"]["stim_dt"] # how much to pad after offset

data = io.load_crcns(cell, config["data"]["stim_type"], config["data"]["root"], 
                     config["strf"]["spec_window"], config["strf"]["stim_dt"], 
                     f_min=config["strf"]["f_min"], 
                     f_max=config["strf"]["f_max"], f_count=config["strf"]["nfreq"], 
                     compress=config["strf"]["spec_compress"], 
                     gammatone=config["strf"]["gammatone"])
io.pad_stimuli(data, config["data"]["pad_before"], pad_after, fill_value=0.0)
io.preprocess_spikes(data, config["mat"]["model_dt"], config["mat"]["taus"])

n_test = int(config["data"]["p_test"] * len(data))

# split into assimilation and test sets and merge stimuli
assim_data = io.merge_data(data[:-n_test])
test_data = io.merge_data(data[-n_test:])
stim = assim_data["stim"]


# ## Estimate parameters
# 
# The reg_alpha and reg_lambda parameters set the L1 and L2 penalties for the
# initial ML estimation. Note that we supply the nonlinearity function to the
# constructor too, as this determines how the log-likelihood is calculated.

# initial guess of parameters using penalized ML. Note that we provide the
# cosine basis set to the constructor of mle.estimator, which causes the design
# matrix to be in the cosine basis set we'll do an initial fit with some strong
# regularization to see if there's an RF. Leave this out of production NB: This
# cell sometimes fails in initializing the estimator; just run it again.

try:
  mlest = mle.mat(assim_data["stim"], kcosbas, assim_data["spike_v"], assim_data["spike_h"],
                assim_data["stim_dt"], assim_data["spike_dt"], nlin=config["mat"]["nlin"])

except:
  mlest = mle.mat(assim_data["stim"], kcosbas, assim_data["spike_v"], assim_data["spike_h"],
                assim_data["stim_dt"], assim_data["spike_dt"], nlin=config["mat"]["nlin"])

w0 = mlest.estimate(reg_lambda=1e1, reg_alpha=1e1)

# The regularization parameters (L1/L2 ratio and total penalty) are chosen using
# cross-validation.

import progressbar
from dstrf import crossvalidate

#reg_grid = np.logspace(-1, 5, 50)[::-1]
l1_ratios = [0.1, 0.5, 0.9] #[0.1, 0.5, 0.7, 0.9, 0.95]
reg_grid = np.logspace(-1, 5, 20)[::-1]

bar = progressbar.ProgressBar(max_value=2 * len(l1_ratios) * len(reg_grid),
                              widgets=[
                                ' [', progressbar.Timer(), '] ',
                                progressbar.Bar(),
                                ' (', progressbar.ETA(), ') ',
                            ])
i = 0
scores = []
results = []
for krank in (1, 2):
    mlest = mle.matfact(assim_data["stim"], kcosbas, krank, assim_data["spike_v"], assim_data["spike_h"],
                        assim_data["stim_dt"], assim_data["spike_dt"], nlin=config["mat"]["nlin"])
    for reg, s, w in crossvalidate.elasticnet(mlest, 4, reg_grid, l1_ratios, avextol=1e-5, disp=False):
        i += 1
        bar.update(i)
        scores.append(s)
        results.append((reg, krank, s, w))
    
best_idx = np.argmax(scores)
best = results[best_idx]

krank = best[1]
rf_alpha, rf_lambda = best[0]
w0 = best[3]
print("best solution: rank={:.3f}, alpha={:.3f}, lambda={:.3f}, loglike={:.3f}".format(krank, rf_alpha, rf_lambda, best[2]))
print(w0[:3])

mlest = mle.matfact(assim_data["stim"], kcosbas, krank, assim_data["spike_v"], assim_data["spike_h"],
                        assim_data["stim_dt"], assim_data["spike_dt"], 
                        nlin=config["mat"]["nlin"])

mltest = mle.matfact(test_data["stim"], kcosbas, krank, test_data["spike_v"], test_data["spike_h"],
                     test_data["stim_dt"], test_data["spike_dt"], nlin=config["mat"]["nlin"])

n_ppost = 10
mat.random_seed(1)
V = mltest.V(w0)
pred = np.zeros_like(test_data["spike_v"])
for i in range(n_ppost):
    pred[:, i] = mltest.predict(w0, mat_fixed, V)
pred_psth = spikes.psth(pred, upsample, 1)
test_psth = spikes.psth(test_data["spike_v"], upsample, 1)

psth_corr = np.corrcoef(test_psth, pred_psth)[0, 1]
eo = performance.corrcoef(test_data["spike_v"][:,::2], test_data["spike_v"][:,1::2], upsample, 1)
print("elastic net penalized maximum likelihood:")
print("loglike: {:.3f}".format(-mltest.loglike(w0)))
print("CC: {:.3f} / {:.3f} ({:.3f})".format(psth_corr, eo, psth_corr/eo))
print("spike count: data = {}, pred = {}".format(test_data["spike_v"].sum() / config["data"]["n_trials"], pred.sum() / n_ppost))
print("\n")

# We'll use the ML estimate to seed the MCMC sampler. We're going to reduce the
# size of the parameter space by factorizing the RF (i.e., a bilinear
# approximation). Note that we try to use the mlest object as much as possible
# to do the calculations rather than reimplement things; however, there can be
# some significant performance enhancements from an optimized implementation.

# estimate parameters using emcee
from neurofit import priors, costs, utils, startpos

# the MAT parameters are just bounded between reasonable limits. These may need
# to be expanded when using real data"
mat_prior = priors.joint_independent(
                [ priors.uniform(config["mat"]["bounds"][0][0], config["mat"]["bounds"][0][1]),
                  priors.uniform(config["mat"]["bounds"][1][0], config["mat"]["bounds"][1][1]),
                  priors.uniform(config["mat"]["bounds"][2][0], config["mat"]["bounds"][2][1]),
                ])

# use the regularization parameters from the cross-validation
rf_alpha, rf_lambda = best[0]

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

np.savez(saveplace + cell + "_" + tag,chain=sampler.flatchain,lnprob=sampler.flatlnprobability,w1=w1,w0=w0,corr=psth_corr,eo=eo)

