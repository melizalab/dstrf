
# coding: utf-8

# # MAT dSTRF Rothman fit

from __future__ import print_function, division
import numpy as np
import scipy as sp
import neurofit as nf
import pyspike as pyspk
from scipy.signal import resample
import cneurons as cn

import sys
sys.path.append("../") # for importing utils and glm

import utils
from models import cosstrf

from neurofit import utils as nfutils

class mat():
    def __init__(self, free_ts=False,stochastic=False):
        self.nrn = None
        self.free_ts = free_ts
        self.stochastic = stochastic

    def dim(self):
        return 5 if self.free_ts else 3

    def set(self, theta):
        a, b, w = theta[:3]
        if self.stochastic:
            self.nrn = cn.augmatsto(a,b,c,w)
        else:
            self.nrn = cn.mat(a, b, w)

        if self.free_ts:
            t1, t2 = theta[3:]
            self.nrn.t1 = t1
            self.nrn.t2 = t2
                
        self.nrn.R = 1
        self.nrn.tm = 1
        
    def run(self, iapp):
        self.nrn.apply_current(iapp, 1)
        return self.nrn.simulate(len(iapp), 1)
    
class dstrf_mat():
    def __init__(self,channels=1,nspec=15,tlen=30,ncos=10,coslin=1,upsample=1,
                 scale=1,free_ts=False,normalize=False,center=False,noise=None,stochastic=False):
        self.mat = mat(free_ts=free_ts,stochastic=stochastic)
        self.pstrf = cosstrf(channels,nspec,tlen,ncos,coslin,normalize,center)
        self.upsample = upsample
        self.scale = scale
        self.free_ts = free_ts
        self.channels = channels
        self.nspec = nspec
        self.tlen = tlen
        self.ncos = ncos
        self.noise = noise
                
    def dim(self):
        return self.channels*(self.nspec+self.ncos) + ( 6 if self.free_ts else 3 )
     
    def set(self, theta):
        cut = -6 if self.free_ts else -3
        self.pstrf.set(theta[:cut])
        self.mat.set(theta[cut:])
        self.mat.nrn.R = 1
        self.mat.nrn.tm = 1
        
    def run(self, stim):
        r = self.pstrf.run(stim)
        if self.noise is not None: r += np.random.randn(len(r))*self.noise
        r = resample(r,len(r)*self.upsample)*self.scale
        return self.mat.run(r)

cell = sys.argv[1]
burn = int(sys.argv[2])
saveplace = sys.argv[3]
tag = sys.argv[4]

print(cell,"\n")

# ## Setting paramters and loading data


# assimilation parameters
nwalkers = 1000
threads = 8
keep = 1
num_assim_stims = 15
tracker = nfutils.convergence_tracker(burn,burn/10)

# dstrf model settings
free_ts = False
scale = 10
channels = 1
ncos = 10
coslin = 1
norm = True
center = True

# data parameters 
nspec = 30
t_dsample = 5
tlen = int(np.rint(150/t_dsample))
psth_smooth = 5/t_dsample
compress = 1

# setup cosine basis
tbas, fromt, tot = utils.cosbasis(tlen,ncos,coslin,retfn=True,norm=True)

# load data
stims,durations,spikes_data,spiky_data = utils.load_rothman(cell+"/",nspec,t_dsample,compress)
psth_data = [utils.psth(spk,dur,t_dsample,dsample=t_dsample) for spk,st,dur in zip(spikes_data,stims,durations)]

# separate the simulation and validation sets
assim_psth, test_psth = np.split(psth_data,[num_assim_stims])
assim_spikes, test_spikes = np.split(spikes_data,[num_assim_stims])
assim_spiky, test_spiky = np.split(spiky_data,[num_assim_stims])
assim_stims, test_stims = stims[:num_assim_stims], stims[num_assim_stims:]
assim_dur, test_dur = np.split(durations,[num_assim_stims])

# calculate correlation between even and odd trial psths
eocorr = [utils.evenoddcorr(spks,dur,dsample=t_dsample,smooth=psth_smooth) for spks,dur in zip(spikes_data,durations)]
print("EO: {:.2f}".format(np.mean(eocorr)))


# ## Getting initial guess for STRF

# In[257]:

# estimate STRF using elastic net regression
fit_psth = [p*1000 for p in assim_psth]
#fit_psth = [normalize(p,False) for p in assim_psth]

STRF_GUESS, B_GUESS = utils.get_strf(assim_stims,fit_psth,tlen,fit_intercept=False)
SPEC,TIM = utils.factorize(STRF_GUESS,channels)

# create initial paramter vector from estimated strf
filt_start = np.hstack((SPEC.flatten(),tot(TIM).flatten()))


# In[258]:

strf_model = cosstrf(channels,nspec,tlen,ncos,coslin,normalize=norm,center=center)
strf_model.set(filt_start)

print("\nSTRF corr:",utils.evaluate(strf_model.filt,test_stims,test_psth))


# ## Define loss and prior functions

# In[259]:

from neurofit import priors
from neurofit import costs

def spike_distance(predict,data):
    trace, spikes = predict
    spiky = pyspk.SpikeTrain(spikes,[0,data[0].t_end])
    dist = 1000*np.mean([pyspk.spike_distance(spiky,trial) for trial in data])
    return dist

mat_prior = priors.joint_independent(
                [ nf.priors.uniform( -300,300),
                  nf.priors.uniform( -10, 10),
                  nf.priors.uniform( -30, 30)])

cost = spike_distance
unbounded = priors.unbounded()

def l1_prior(theta):
    return -np.sum(np.abs(theta))

def dstrf_prior(theta):
    return l1_prior(theta[:-3]) + mat_prior(theta[-3:])


# ## Run initial MAT parameter fit

# In[260]:

from neurofit import startpos

# get I with STRF fixed
Iapp = []
for s,dur in zip(stims,durations):
    R = resample(strf_model.run(s),dur)
    Iapp.append(R*scale)
    
assim_Iapp, test_Iapp = np.split(Iapp,[num_assim_stims])

# initalize the mat model
mat_model = mat(free_ts=free_ts)

# generate starting positions of emcee walkers
p0 = startpos.uniform_independent(nwalkers,[-300,-10,-10],[300,10,10])
#p0 = startpos.normal_independent(nwalkers,[5,0,0,1],[0.1]*4)
#p0 = startpos.normal_independent(nwalkers,matparam,[0.1]*4)

# run emcee
mat_smplr = nf.sampler(mat_model,mat_prior,cost,nwalkers,zip(assim_Iapp,assim_spiky),threads)

print("\n")
for pos,_,_ in tracker(mat_smplr.sample(p0,iterations=burn)): continue
mat_smplr.reset()
mat_smplr.run_mcmc(pos,1);


# In[261]:

# check the performance of the fit mat model
mml = mat_smplr.flatchain[np.argmax(mat_smplr.flatlnprobability)]
mat_map = mat()
mat_map.set(mml)
mat_corr = []

param_corr = utils.evaluate(STRF_GUESS,test_stims,test_psth)

for i,p,d in zip(test_Iapp,test_psth,test_dur):
    trace,spikes = mat_map.run(i)
    mat_psth = utils.psth_spiky(pyspk.SpikeTrain(spikes,[0,d]),binres=1,smooth=psth_smooth,dsample=t_dsample)
    mat_corr.append(np.corrcoef(p,mat_psth)[0][1])
        
start = np.hstack((filt_start,mml))
print("\nFilt R: {:.3f}, MAT R: {:.3f}".format(param_corr,np.mean(mat_corr)))


# ## Fit dSTRF model using emcee

# In[262]:

# initalize the model
model = dstrf_mat(channels,nspec,tlen,ncos,coslin,t_dsample,scale=scale,normalize=norm,center=center)

# set starting positions for walkers
p01 = startpos.normal_independent(nwalkers,filt_start,[1e-1]*len(start))
p02 = mat_smplr.chain[:,0,:]
p0 = np.hstack((p01,p02))

# run emcee
print("\n")
dstrf_smplr = nf.sampler(model,dstrf_prior,spike_distance,nwalkers,zip(assim_stims,assim_spiky),threads)
for pos,_,_ in tracker(dstrf_smplr.sample(p0,iterations=burn)): continue
dstrf_smplr.reset()
dstrf_smplr.run_mcmc(pos,1);


# ## Evaluate the model fit

# In[263]:

# initalize model with MAP parameter estimate
dmap = dstrf_smplr.flatchain[np.argmax(dstrf_smplr.flatlnprobability)]
model.set(dmap)


# In[264]:

map_corr, map_psths = utils.dstrf_sample_validate(model,dmap,test_stims,test_psth,t_dsample,psth_smooth)
ppcorr, pp_psths = utils.posterior_predict_corr(model,test_stims,test_psth,dstrf_smplr.flatchain,t_dsample,psth_smooth)
corr_means = np.mean([map_corr,ppcorr,eocorr[num_assim_stims:]],axis=1)
print("\nMAP: {:.2f}, Dist: {:.2f}, EO: {:.2f}".format(corr_means[0],corr_means[1],corr_means[2]))
print("MAP/EO: {:.2f}, Dist/EO: {:.2f}".format(corr_means[0]/corr_means[2],corr_means[1]/corr_means[2]))

np.savez(saveplace + cell + "_" + tag,chain=dstrf_smplr.flatchain,lnprob=dstrf_smplr.flatlnprobability,map=dmap)
