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
from scipy import signal as ss
import time

import yaml
import pickle

import mat_neuron._model as mat
from dstrf import strf, mle, io, performance, spikes
cell = sys.argv[1]
yfile = sys.argv[2]
saveplace = sys.argv[3]
tag = sys.argv[4]

interfile = saveplace + "/" + cell + "_" + tag + "-ml.dat"

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

# load stimuli and responses
data = io.load_rothman(cell,config["data"]["root"],
                     config["strf"]["spec_window"],
                     config["strf"]["stim_dt"],
                     f_min=config["strf"]["f_min"],
                     f_max=config["strf"]["f_max"], f_count=50,
                     compress=config["strf"]["spec_compress"],
                     gammatone=config["strf"]["gammatone"])
io.pad_stimuli(data, config["data"]["pad_before"], pad_after, fill_value=0.0)
io.preprocess_spikes(data, config["mat"]["model_dt"], config["mat"]["taus"])

n_test = int(config["data"]["p_test"] * len(data))

# split into assimilation and test sets and merge stimuli
assim_data = io.merge_data(data[:-n_test])
test_data = io.merge_data(data[-n_test:])

assim_data["stim"] = ss.resample(assim_data["stim"],config["strf"]["nfreq"])
test_data["stim"] = ss.resample(test_data["stim"],config["strf"]["nfreq"])

eo = performance.corrcoef(test_data["spike_v"][:,::2], test_data["spike_v"][:,1::2], upsample, 1)

if eo < 0.2:
  print("Even/Odd Correlation too low: {:.3f}".format(eo))
  sys.exit(0)

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

START = time.time()

l1_ratios = config["crossval"]["l1_ratios"]
reg_grid = np.logspace(config["crossval"]["logspace_start"], config["crossval"]["logspace_end"],
                       config["crossval"]["logspace_steps"])[::-1]

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

FINISH = time.time()

HRS,MINS = divmod(FINISH-START,60)

print("cross-validation took {}:{:02d}".format(HRS, int(MINS)))

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
print("elastic net penalized maximum likelihood:")
print("loglike: {:.3f}".format(-mltest.loglike(w0)))
print("CC: {:.3f} / {:.3f} ({:.3f})".format(psth_corr, eo, psth_corr/eo))
print("spike count: data = {}, pred = {}".format(test_data["spike_v"].sum() / config["data"]["n_trials"], pred.sum() / n_ppost))
print("\n")

with open(interfile, 'wb') as outfile:
     pickle.dump(dict(file=interfile,
                      yfile=yfile,
                      cell=cell,
                      w0=w0,
                      assim_data=assim_data,
                      test_data=test_data,
                      rf_lambda=rf_lambda,
                      rf_alpha=rf_alpha,
                      krank=krank,
                      mlest=mlest,
                      mltest=mltest,
                      corr=psth_corr,
                      eo=eo),
                  outfile,protocol=pickle.HIGHEST_PROTOCOL)
