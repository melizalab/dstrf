
# coding: utf-8

from __future__ import print_function, division
import numpy as np
import scipy as sp
import neurofit as nf
import pyspike as pyspk
from scipy.signal import resample

import sys
sys.path.append("../") # for importing utils and models

import utils
import cneurons as cn
from models import cosstrf, GLM_cos

from neurofit import utils as nfutils

cell = sys.argv[1]
burn = int(sys.argv[2])
saveplace = sys.argv[3]
tag = sys.argv[4]

print(cell,"\n")

# ## Setting paramters and loading data

# In[2]:

# assimilation parameters
nwalkers = 1000
threads = 8
keep = 1
num_assim_stims = 15
tracker = nfutils.convergence_tracker(burn,burn/10)

# glm model settings
ncos = 10
spcos = 5
dt = 0.001
tcoslin = 1
hcoslin = 10
channels = 1

# data parameters 
nspec = 30
t_dsample = 5
tlen = int(np.rint(150/t_dsample))
plen = int(np.rint(50/t_dsample))
psth_smooth = 5/t_dsample
compress = 1

# setup cosine basis
tbas, fromt, tot = utils.cosbasis(tlen,ncos,tcoslin,retfn=True,norm=True)
hbas, fromh, toh = utils.cosbasis(plen,spcos,hcoslin,retfn=True)

# load data
stims,durations,spikes_data,spiky_data = utils.load_rothman(cell+"/",nspec,t_dsample,compress)
psth_data = [utils.psth(spk,dur,t_dsample,dsample=t_dsample) for spk,st,dur in zip(spikes_data,stims,durations)]

# calculate correlation between even and odd trial psths
eocorr = [utils.evenoddcorr(spks,dur,dsample=t_dsample,smooth=psth_smooth) for spks,dur in zip(spikes_data,durations)]
print("EO: {:.2f}".format(np.mean(eocorr)))

spikes_data= [[(np.round(trial)/t_dsample).astype(int) for trial in s] for s in spikes_data]
durations = [int(np.rint(d/t_dsample)) for d in durations]

# separate the simulation and validation sets
assim_psth, test_psth = np.split(psth_data,[num_assim_stims])
assim_spikes, test_spikes = np.split(spikes_data,[num_assim_stims])
assim_spiky, test_spiky = np.split(spiky_data,[num_assim_stims])
assim_stims, test_stims = stims[:num_assim_stims], stims[num_assim_stims:]
assim_dur, test_dur = np.split(durations,[num_assim_stims])


# ## Getting initial guess for STRF

# In[3]:

# estimate STRF using elastic net regression
fit_psth = [np.log(p*1000 + 1) for p in assim_psth]

STRF_GUESS, B_GUESS = utils.get_strf(assim_stims,fit_psth,tlen,fit_intercept=True)
SPEC,TIM = utils.factorize(STRF_GUESS,1)

# create initial paramter vector from estimated strf
filt_start = np.hstack(([B_GUESS],SPEC.flatten(),tot(TIM).flatten()))
strf_model = cosstrf(channels,nspec,tlen,ncos,tcoslin)
strf_model.set(filt_start[1:])

print("\nSTRF corr: ",utils.evaluate(strf_model.filt,test_stims,test_psth),"\n")


# ## Define loss and prior functions

# In[5]:

from neurofit import priors
from neurofit import costs
from neurofit import startpos


def pploss(predict,data):
    lam,spikes = predict
    if np.shape(lam[0]) == ():
        lam = [lam]
        spikes = [spikes]
    return -np.sum([np.sum(np.log(l[s])) - np.sum(l) for s,l in zip(spikes,lam)])

def l1_prior(theta):
    return -np.sum(np.abs(theta))

def cost(theta, model, lnprior, lnlike, observs,fixed):

    lp = lnprior(theta)
    if not np.isfinite(lp): return -np.inf
    params = np.hstack((fixed,theta))
    model.set(params)
    
    ll = 0
    for stim, data, in observs:
        ll += lnlike(model.run(stim), data)
        if np.isnan(ll): return -np.inf
        
    return -ll + lp


# ## Run initial post spike filter fit

# In[6]:

# initalize the model
model = GLM_cos(channels,nspec,tlen,plen,ncos,spcos,tcoslin,hcoslin,nonlin=np.exp,spike=True,dt=dt)

data = zip(zip(test_stims,test_spikes),test_psth)

ncost = lambda *args: -cost(*args)

from scipy import optimize as opt

fixed = filt_start
result = opt.minimize(ncost,[0]*5,args=(model,priors.unbounded(),pploss,data,fixed))
start = np.hstack((fixed,result['x']))

# ## Fit GLM model using emcee

# In[8]:

# set starting positions for walkers
p0 = startpos.normal_independent(nwalkers-1,start,[1e-4]*len(start))
p0 = np.vstack((start,p0))

# run emcee
glm_smplr = nf.sampler(model,l1_prior,pploss,nwalkers,zip(zip(assim_stims,assim_spikes),test_psth),threads)
for pos,_,_ in tracker(glm_smplr.sample(p0,iterations=burn)): continue
glm_smplr.reset()
glm_smplr.run_mcmc(pos,1);


# ## Evaluate the model fit

# In[287]:

# initalize model with MAP parameter estimate
dmap = glm_smplr.flatchain[np.argmax(glm_smplr.flatlnprobability)]
model.set(dmap)

# In[283]:

map_corr = utils.glm_sample_validate(model,dmap,test_stims,test_psth,ntrials=10,dsample=1,smooth=psth_smooth)
ppcorr = utils.glm_post_predict_corr(model,test_stims,test_psth,glm_smplr.flatchain,1,psth_smooth)
corr_means = np.mean([map_corr,ppcorr,eocorr[num_assim_stims:]],axis=1)
print("\nMAP: {:.2f}, Dist: {:.2f}, EO: {:.2f}".format(corr_means[0],corr_means[1],corr_means[2]))
print("MAP/EO: {:.2f}, Dist/EO: {:.2f}".format(corr_means[0]/corr_means[2],corr_means[1]/corr_means[2]))

np.savez(saveplace + cell + "_" + tag,chain=glm_smplr.flatchain,lnprob=glm_smplr.flatlnprobability,map=dmap)