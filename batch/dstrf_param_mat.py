# coding: utf-8

from __future__ import division
from __future__ import print_function

import numpy as np
import emcee
import pyspike as spk
import neurofit as nf
import neurofit.cneurons as cn

from scipy.signal import decimate, resample

import utils
import os
import shutil
import time
from os import walk
import sys

cell = sys.argv[1]
stim_type = sys.argv[2]

saveplace = sys.argv[3]

tag = sys.argv[4]

save_folder = cell + "-" + stim_type + "_" + "dstrf_param_mat"

pbar = True

# dstrf settings
nwalkers = 1000
burn = 500
dstrf_burn = burn*2
threads = 8
keep = 1
num_assim_stims = 15
nbatch = 15
free_ts = False

nspec = 30
t_dsample = 5
tlen = int(np.rint(150/t_dsample))
psth_smooth = 5/t_dsample

compress = 1
nonlin = False


channels = 3
scale = 1



mat_bounds  = [[-1000, 1000],
               [ -50,  50],
               [-100, 100],
               [ -30,  30]]


if free_ts: mat_bounds.extend([[1,100],[100,500]])

tlen_bounds  = [[    5,   150]]
param_bounds = [[ -100,   100],
                [ -100,   100],
                [ -100,   100],
                [ -100,   100],
                [    0,    50],
                [    0, nspec],
                [    0, nspec]]

bounds = mat_bounds + tlen_bounds + param_bounds*channels

if free_ts:
    bounds[4] = [0,500]
    bounds[5] = [0,500]

# load data from crcns
spikesroot = "/home/data/crcns/all_cells/" + cell + "/" + stim_type +"/"
stimroot = "/home/data/crcns/all_stims/"

spikefiles = [f for f in next(walk(spikesroot))[2] if len(f.split(".")) == 2]
stimfiles = [f.split(".")[0] + ".wav" for f in spikefiles]

stims,durations = utils.load_sound_data(stimfiles, stimroot, dsample=t_dsample,sres=nspec,gammatone=True,compress=compress)
spikes_data, spiky_data = utils.load_spikes_data(spikefiles, spikesroot, durations)
psth_data = [utils.psth(spiky,binres=1,smooth=psth_smooth,dsample=t_dsample) for spiky in spiky_data]
norm_psth_data = [nf.utils.normalize(p,center=False) for p in psth_data]
psth_iapp = [resample(nf.utils.normalize(x),durations[i])*scale for i,x in enumerate(psth_data)]
norm_psth_data = [nf.utils.normalize(p) for p in psth_data]

# define dstrf class
class dstrf_prior:
    def __init__(self,bounds,free_ts=False):
        self.bounds = bounds
        self.free_ts = free_ts
    
    def __call__(self,theta):
        for i, x in enumerate(theta):
            if (x < self.bounds[i][0] or x > self.bounds[i][1]):
                return -np.inf
        if free_ts and (theta[5] < theta[4]):
            return -np.inf
        return 0.0

class dstrf():
    def __init__(self,channels=1,nspec=15,upsample=1,scale=1,free_ts=False,nonlin=False):
        self.mat = nf.models.mat(free_ts=free_ts)
        self.pstrf = nf.models.parameterized_strf(channels,nspec,nonlin=nonlin)
        self.upsample = upsample
        self.scale = scale
        self.free_ts = free_ts
    
    def set(self, theta):
        cut = 6 if self.free_ts else 4
        self.mat.set(theta[:cut])
        self.pstrf.set(theta[cut:])
        
    def run(self, stim):
        r = self.pstrf.run(stim)
        r = resample(nf.utils.normalize(r),len(r)*self.upsample)*self.scale
        return self.mat.run(r)

## Running the Sampler
print("Running initial fits...")
begin = time.time()

param_fit = nf.examples.fit_parameterized_strf(norm_psth_data[:num_assim_stims],stims[:num_assim_stims],
                                               channels=channels,nonlin=nonlin,
                                               burn=burn,progress_bar=pbar,threads=threads)
pml = param_fit.flatchain[np.argmax(param_fit.flatlnprobability)]
param_maxlik = nf.models.parameterized_strf(channels,nspec)
param_maxlik.set(pml)
param_corr = utils.evaluate(param_maxlik.filt,stims,psth_data)

I_param = []
for i,x in enumerate(stims):
    I_param.append(resample(param_maxlik.run(x),durations[i])*scale)

mat_fit = nf.examples.fit_mat(spiky_data[:num_assim_stims],I_param[:num_assim_stims],burn=burn,threads=threads,progress_bar=pbar,free_ts=free_ts)
mml = mat_fit.flatchain[np.argmax(mat_fit.flatlnprobability)]
mat_maxlik = nf.models.mat()
mat_maxlik.set(mml)
end = time.time()

print("Sampling finished. Took {} minutes.".format((end - begin)/60))
param_corr = utils.evaluate(param_maxlik.filt,stims[num_assim_stims:],psth_data[num_assim_stims:])
mat_corr = []
for iapp,data in zip(I_param[num_assim_stims:],psth_data[num_assim_stims:]):
    trace,spikes = mat_maxlik.run(iapp)
    dur = len(iapp)
    mat_psth = utils.psth(spk.SpikeTrain(spikes,[0,dur]),binres=1,smooth=psth_smooth,dsample=t_dsample)
    mat_corr.append(np.corrcoef(data,mat_psth)[0][1])
    
print("Filt R: {:.3f}, MAT R: {:.3f}".format(param_corr,np.mean(mat_corr)))

print("param maxlik:")
print(pml)

print("MAT maxlik:")
print(mml)

print("Fitting dstrf...")
start = np.hstack((mml,pml))

out = nf.fit(spiky_data,stims,
             dstrf(channels=channels,nspec=nspec,upsample=t_dsample,scale=scale,free_ts=free_ts),
             dstrf_prior(bounds,free_ts=True),nf.examples.spiky_hinge,
             nwalkers,dstrf_burn,keep,
             nf.startpos.gaussian(start,[0.1]*len(start),nwalkers),
             progress_bar=pbar,threads=threads)

end = time.time()

print("Sampling finished. Took {} minutes.".format((end - begin)/60))

maxlik = out.flatchain[out.flatlnprobability.argmax()]

print("MAP:")
print(maxlik)

maxlik_nrn = dstrf(channels=channels,nspec=nspec,upsample=t_dsample,scale=scale,free_ts=free_ts) 
maxlik_nrn.set(maxlik)

dstrf_corr = []
for stim, data in zip(stims,psth_data)[num_assim_stims:]:
    trace,spikes = maxlik_nrn.run(stim)
    dur = len(data)*t_dsample
    mat_psth = utils.psth(spk.SpikeTrain(spikes,[0,dur]),binres=1,smooth=psth_smooth,dsample=t_dsample)
    dstrf_corr.append(np.corrcoef(data,mat_psth)[0][1])

print("dSTRF R: {:.3f}".format(np.mean(dstrf_corr)))
np.savez(saveplace + cell + "_" + stim_type + "_" + tag,chain=out.flatchain,lnprob=out.flatlnprobability,maxlik=maxlik)
