from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import pyspike as spk
import sklearn.linear_model as lm

from scipy.signal import decimate, resample, resample_poly
import scipy.ndimage.filters as sf

from scipy.io.wavfile import read 

import toelis as ts

import libtfr
import gammatone.gtgram as gg

from os import walk


def spgconv(h,s,pad="edge"):
    npix, nts = np.shape(s)
    __, tlen = np.shape(h)
        
    plen = tlen-1
    end = nts-plen if not pad else nts
    
    s = s if not pad else np.pad(s,((0,0),(plen,0)),pad)
    
    cs = np.zeros(end)
    for i in range(npix):
        cs += np.convolve(s[i,:],h[i,:],'valid')
    return cs 

def strf(resolution=50,time=50,maxfreq=8000,latency=0,frequency=0,A=0.25,sigma=0.1,gamma=0.001,alpha=1.4,beta=1.5):
    import numpy as np    
    time -= 1
    scale = resolution/50.0   
    t = np.arange(float(np.negative(time)),1)
    tscale = np.arange(np.negative(time),1,2)
    x = latency
    f = np.arange(0,maxfreq+1,float(maxfreq)/resolution)
    y = frequency
    tc = t+x
    fc = f-y
    tprime, fprime = np.meshgrid(tc,fc)
    sigma = sigma/scale
    Gtf = A*np.exp(-sigma**2*tprime**2-gamma**2*fprime**2)*(1-alpha**2*sigma**2*tprime**2)*(1-beta**2*gamma**2*fprime**2)
    return (Gtf,tscale,f)

def design_matrix(stims,rs,twidth):
    X = np.vstack([np.hstack([sp.linalg.hankel(f[:-twidth+2],f[-twidth:]) for f in s]) for s in stims])
    R = np.hstack(r[twidth-2:] for r in rs)
    return X,R

def ps_design(stims,rs,pwidth,twidth):
    X = np.vstack([sp.linalg.hankel(s[twidth-pwidth-1:-pwidth],s[-pwidth:]) for s in stims])
    R = np.hstack(r[twidth-1:] for r in rs)    
    return X,R

def get_strf(stims,rs,twidth,mode="ElasticNetCV",normalize=True,**kwargs):

    nspec = stims[0].shape[0]
    X,R = design_matrix(stims,rs,twidth)

    regress = getattr(lm,mode)(normalize=normalize,**kwargs)
    regress.fit(X,R)
    
    return np.fliplr(regress.coef_.reshape(nspec,twidth)), regress.intercept_

def get_strf_ps(stims,rs,twidth,pwidth,spikes=None,mode="ElasticNetCV",normalize=True,**kwargs):
  
    spikes = rs if spikes == None else spikes

    nspec = stims[0].shape[0]
    S,R = design_matrix(stims,rs,twidth)
    P   = ps_design(spikes,rs,pwidth,twidth)[0]

    X = np.hstack((S,P))

    regress = getattr(lm,mode)(normalize=normalize,**kwargs)
    regress.fit(X,R)

    strf, post = np.split(regress.coef_,[-pwidth])
    strf = np.fliplr(strf.reshape(nspec,twidth))

    post = post[::-1]
    
    return strf, post, regress.intercept_

def get_strf_old(stims,rs,twidth,smooth=0,thresh=None,offset=True):

    X = np.vstack([np.hstack([sp.linalg.hankel(f[:-twidth+1],f[-twidth:]) for f in s]) for s in stims])
    R = np.hstack(r[twidth-1:] for r in rs)

    nspec = stims[0].shape[0]

    if offset:
        ones = np.ones((len(X),1))
        X = np.hstack((X,ones))

    covX = np.matmul(X.T,X)
    STA = np.matmul(X.T,R)
    params = np.matmul(np.linalg.pinv(covX),STA)

    if offset: params, bias = np.split(params,[-1])

    strf = params.reshape(nspec,twidth)[:,::-1] 

    if smooth: strf = sf.gaussian_filter(strf,smooth)
    if thresh: strf[np.abs(strf)/np.abs(strf.max())<thresh] = 0
    return strf, bias[0] if offset else strf

def evaluate(h,stims,rs,offset=0,nonlin=None):
    if nonlin is None: nonlin = lambda x: x
    corcof = 0
    nstim = len(stims)
    for i, stim in enumerate(stims):
        R = nonlin(spgconv(np.asarray(h),stim) + offset)
        corcof += np.corrcoef(rs[i],R)[0][1]
    return corcof/nstim

def psth(spikes,duration,smooth,dsample=0):
    ntrains = len(spikes)
    mz = np.zeros(duration)
    for s in spikes:
        s = map(int,np.round(np.asarray(s)))
        z = np.zeros(duration+1)
        z[s] = 1
        mz += z[:-1]
    mz = sf.gaussian_filter1d(mz,smooth,mode="constant")
    mz /= (ntrains/smooth) if smooth != 0 else ntrains
    if dsample > 1:
        #mz = np.interp(np.arange(0,duration,dsample),np.arange(duration),mz)
        mz = resample(mz,int(duration/dsample))
    mz[mz<0] = 0
    return mz

def psth_spiky(spiky,binres=1,dsample=False,smooth=False):
    if np.shape(spiky[0]) == (): spiky = [spiky]
    ntrains = len(spiky)
    raw_psth = spk.psth(spiky,binres).y
    psth = raw_psth/(ntrains*binres)
    if dsample: psth = resample(psth,int(len(psth)/dsample))
    if smooth: psth = sf.gaussian_filter1d(psth,smooth,mode="constant")
    return psth
    
def SNR(spiky_data,bin_length=1,smooth=10,dsample=0):

    PSTH = [psth(spky,bin_length,smooth,dsample) for i, spky in enumerate(spiky_data)]
    R = []
    A = []

    for i, trial1 in enumerate(PSTH):
        for j, trial2 in enumerate(PSTH):
            if i == j: R.append(np.cov(trial1,trial2))
            else: A.append(np.cov(trial1,trial2)[0][1])
                
    mR = np.mean(R)
    mA = np.mean(A)

    return mA/(mR-mA)

def corrnorm(spiky_data,bin_length=1,smooth=1,dsample=10):
    if np.shape(spiky_data) == (): spiky_data = [spiky_data]
    PSTH = [psth(spky,bin_length,smooth=smooth,dsample=dsample) for i, spky in enumerate(spiky_data)]
    A = []

    for i, trial1 in enumerate(PSTH):
        for j, trial2 in enumerate(PSTH):
            if i == j: pass
            else: A.append(np.corrcoef(trial1,trial2)[0][1])
    mA = np.mean(A)
    
    return np.sqrt(mA)   

def tbt_corr(predict, spiky_data, binres=1, dsample=10, smooth=1):
    ntrains = len(spiky_data)
    corr = 0
    for spiky in spiky_data:
        corr += np.corrcoef(predict,psth(spiky,binres,dsample,smooth))[0][1]
    return corr/ntrains
    

def load_sound_data(files,root="",windowtime=256,ovlerlap=10,f_min=500,f_max=8000,gammatone=False,
                    dsample=10,sres=15,compress=0):
    stims = []
    durations = []

    for f in files:
        Fs, wave = read(root+f)
        wt = windowtime/Fs
        ovl = ovlerlap/Fs

        duration = int(1000*len(wave)/Fs)
        durations.append(duration)
        if gammatone:
            Pxx = gg.gtgram(wave,Fs,wt,ovl,sres,f_min,f_max)
            Pxx = 10*np.log10(Pxx+compress)

        else:
            w = np.hanning(int(windowtime))
            Pxx = libtfr.stft(wave, w, int(w.size * .1))
            freqs, ind = libtfr.fgrid(Fs, w.size, [f_min, f_max])
            Pxx = Pxx[ind,:]
            Pxx = 10*np.log10(Pxx+compress)
            Pxx = resample(Pxx,sres)
        Pxx = resample(Pxx,int(duration/dsample),axis=1)
        stims.append(Pxx)
    return stims,durations

def load_spikes_data(files,root="",durations=[],ts_file=True,delim=" "):
    spikes_data = []
    spiky_data = []
    for d,f in enumerate(files):
        # load spike data
        dur = durations[d]
        if ts_file: stim_spikes = [filter(lambda x: x >= 0 and x <= dur, s) for s in ts.read(open(root + f))[0]]
        else: stim_spikes = [filter(lambda x: x >= 0 and x <= dur, map(float,s.split(delim))) for s in open(root + f)]
        spikes_data.append(stim_spikes)
        stim_spiky = [spk.SpikeTrain(s,[0,durations[d]]) for s in stim_spikes]
        spiky_data.append(stim_spiky)
        
    return spikes_data, spiky_data

def prune_walkers(pos,lnprob,tolerance=10,resample=None,return_indx=False,cutoff_exclude=10):
    mean_lnprob = np.mean(lnprob,axis=1)
    sorted_mean = np.sort(mean_lnprob)
    gradient = np.gradient(sorted_mean)[:-cutoff_exclude]
    
    cutoff_indx = np.where(gradient>np.mean(gradient)*tolerance)[0][-1]
    cutoff = sorted_mean[cutoff_indx]
    prune = np.where(mean_lnprob > cutoff)[0]
   
    if resample: prune = np.random.choice(prune,resample)
    if return_indx: return pos[prune], prune
    else: return pos[prune] 
    
def P3Z1(t,z1,p1,p2,p3,l,A):
    zeros = (t-z1)
    poles = (t+p1)*(t+p2)*(t+p3)
    out = A*np.exp(-l*t)*poles/zeros
    return out if np.isfinite(out) else 0 
    
def PZ(t,zs,ps,l,A):
    zeros = 1
    poles = 1
    for z in zs: zeros *= (t-z)
    for p in ps: poles *= (t+p)
    out = A*np.exp(-l*t)*poles/zeros
    return out if np.isfinite(out) else 0 

def gauss(x, mu=0, sig=1):
    return np.exp(-np.power((x-mu),2)/(2*np.power(sig,2)))/np.sqrt(2*np.power(sig,2)*np.pi)

def morlet(x, mu=0, sig=1, dep=1):
    return np.exp(-np.power((x-mu),2)/(np.power(sig,2)) - 1j*dep*(x-mu))

def overtime(l, u, f, *args):
    out = []
    for a in range(l,u):
        out.append(f(a,*args))
    return out

def factorize(strf,channels=None,min_channels=1):
    sres, tres = np.shape(strf)
    U,s,V = np.linalg.svd(strf)
    
    if not channels: channels = max(min_channels,len(np.where(s>1.0)[0])) 
    
    time = np.ndarray((channels,tres))
    spec = np.ndarray((channels,sres))
    for i in range(channels):
        time[i] = V[i,:]*s[i]
        spec[i] = U[:,i]
    
    return spec, time

def trialcorr(spiky_data,bin_length=1,smooth=1,dsample=10):
    spiky_data = np.asarray(spiky_data)
    A = []
    trialn = range(len(spiky_data))
    for i, trial in enumerate(spiky_data):
        ixclude = [n for n in trialn if n != i]
        ipsth = psth([trial],bin_length,smooth=smooth,dsample=dsample)
        compare = psth(spiky_data[ixclude],bin_length,smooth=smooth,dsample=dsample)
        A.append(np.corrcoef(ipsth,compare)[0][1])
    return np.nanmean(A)

def evenoddcorr(spikes,duration,smooth=1,dsample=10):
    spikes = np.asarray(spikes)
    trialns = np.arange(len(spikes))
    evens = np.where(trialns % 2 == 0)[0]
    odds = np.where(trialns % 2 != 0)[0]
        
    even_psth = psth(spikes[evens],duration,smooth=smooth,dsample=dsample)
    odd_psth = psth(spikes[odds],duration,smooth=smooth,dsample=dsample)
    return np.corrcoef(even_psth,odd_psth)[0][1]

def cosbasis(dur,ncos,lin=1,peaks=None,norm=False,retfn=False):

    peaks = np.asarray([0,  dur*(1-1.5/ncos)]) if peaks is None else np.asarray(peaks)
    nlin = lambda x: np.log(x+1e-20)
    invnl = lambda x: np.exp(x)-1e-20

    y = nlin(peaks+lin)
    db = np.diff(y)/(ncos-1)
    ctrs = np.arange(y[0],y[1]+db,db)
    mxt = invnl(y[1]+2*db)-lin
    kt0 = np.arange(0,mxt)
    nt = len(kt0)
    f = lambda c: (np.cos(np.clip((nlin(kt0+lin)-c)*np.pi/db/2,-np.pi,np.pi))+1)/2

    basis = np.asarray(map(f,ctrs)).T[:dur,:ncos]
    if norm: basis /= np.linalg.norm(basis,axis=0)
    if retfn:
        tobas = lambda v: np.matmul(v,np.linalg.pinv(basis).T)
        frombas = lambda v: np.matmul(basis,np.asarray(v).T).T
        return basis, frombas, tobas
    else: return basis

def evenoddcorr(spikes,duration,smooth=1,dsample=10):
    spikes = np.asarray(spikes)
    trialns = np.arange(len(spikes))
    evens = np.where(trialns % 2 == 0)[0]
    odds = np.where(trialns % 2 != 0)[0]
        
    even_psth = psth(spikes[evens],duration,smooth=smooth,dsample=dsample)
    odd_psth = psth(spikes[odds],duration,smooth=smooth,dsample=dsample)
    return np.corrcoef(even_psth,odd_psth)[0][1]

def load_crcns(cell,stim_type,nspec,t_dsample,compress=1,gammatone=True,root="/home/data/crcns/"):
    spikesroot = root + "/all_cells/" + cell + "/" + stim_type +"/"
    stimroot = root + "/all_stims/"

    spikefiles = [f for f in next(walk(spikesroot))[2] if len(f.split(".")) == 2]
    stimfiles = [f.split(".")[0] + ".wav" for f in spikefiles]

    stims,durations = load_sound_data(stimfiles, stimroot, dsample=t_dsample,sres=nspec,gammatone=gammatone,compress=compress)
    spikes_data, spiky_data = load_spikes_data(spikefiles, spikesroot, durations)
    
    return stims,durations,spikes_data,spiky_data

def posterior_predict_corr(model,stims,data,flatchain,t_dsample=1,psth_smooth=1,nsamples=100,bootstrap=False):

    idx = [None]
    smpl_idx = None
    MANY = []

    for i in range(nsamples):
        while smpl_idx in idx:
            smpl_idx = np.random.randint(len(flatchain))
        if not bootstrap: idx.append(smpl_idx)
        smpl = flatchain[smpl_idx]
        model.set(smpl)

        resp_spiky = []
        for s in stims:
            trace, spikes = model.run(s)
            spky = spk.SpikeTrain(spikes,[0,len(trace)])
            resp_spiky.append(spky)
        MANY.append(resp_spiky)

    corr = []
    for i in range(len(stims)):
        p = psth_spiky(np.asarray(MANY)[:,i],1,t_dsample,psth_smooth)
        thiscorr = np.corrcoef(p,data[i])[0][1]
        corr.append(thiscorr)
    return corr

def dstrf_sample_validate(model,sample,stims,psth,t_dsample=1,psth_smooth=1,plt=True,sscale=0.1,figsize=(16,5)):
    model.set(sample)

    smpl_corr = []
    thresh = None

    for s,p in zip(stims,psth):
        trace, spikes = model.run(s) 
        if np.ndim(trace) == 2: 
            V, thresh = trace.T
            height = max(thresh)
            spad = height/10
            V[spikes] = height
        else: 
            V = trace
            height = max(trace)
            spad = height/10

        spky = spk.SpikeTrain(spikes,[0,len(trace)])
        corr = np.corrcoef(psth_spiky(spky,binres=1,smooth=psth_smooth,dsample=t_dsample),p)[0][1]
        smpl_corr.append(corr)
        
        if plt:
            import matplotlib.pyplot as pyplt
            import seaborn as sns
            clr = sns.color_palette('cubehelix',5)
            
            pyplt.figure(figsize=figsize)
            pyplt.title("$R: {:.2f}$".format(corr))
            if thresh is not None: pyplt.plot(thresh,alpha=0.25,color=clr[1])
            pyplt.plot(V,linewidth=1,color=clr[1])
            for i,trial in enumerate(k):
                pyplt.vlines(trial.spikes,spad+height+sscale*height*i,spad+height+sscale*height*(i+1),alpha=1,color=clr[0])
            i += 1
            pyplt.vlines(spikes,spad+height+sscale*height*(i+sscale),spad+height+sscale*height*(i+1+sscale),alpha=1,color=clr[1])
            pyplt.xlim(0,len(V))
    if plt: pyplt.tight_layout()
    return smpl_corr

def glm_sample_validate(model,sample,stims,psth_data,ntrials=10,smooth=1,dsample=0):
    corrs = []
    for s,p in zip(stims,psth_data):
        trials = [model.run(s)[1] for i in range(ntrials)]
        corrs.append(np.corrcoef(psth(trials,len(p),smooth,dsample),p)[0][1])
    return corrs

def glm_post_predict_corr(model,stims,data,flatchain,t_dsample,psth_smooth,nsamples=100,ntrials=10,bootstrap=False):
    idx = []
    smpl_idx = None
    corrs = []
    for i in range(nsamples):
            while True:
                smpl_idx = np.random.randint(len(flatchain))
                if smpl_idx not in idx: break
            if not bootstrap: idx.append(smpl_idx)
            smpl = flatchain[smpl_idx]
            corrs.append(glm_sample_validate(model,smpl,stims,data,smooth=psth_smooth,ntrials=ntrials,dsample=t_dsample))
    return np.mean(corrs,axis=0)

def specs_to_designs(specs,tlen,pad="edge"):
    dummy = [[]]*len(specs)
    return [design_matrix([np.pad(s,((0,0),(tlen-1,0)),pad)],[p],tlen)[0] for s,p in zip(specs,dummy)]

def spksbin_to_designs(spks,tlen,plen,pad="edge"):
    dummy = [[]]*len(specs)
    return [[ps_design([np.pad(s,(tlen-1,0),pad)],[psth[0]],plen,tlen)[0] for s in ss ] for ss in dummy]

def load_rothman(model,nspec,t_dsample,compress=1,gammatone=True,root="/scratch/mcb2x/modeldata/sigma5/"):
    stimfiles = []
    spikefiles = []
    for f in next(walk(root+model))[2]:
        split = f.split(".")
        if len(split)==2:
            if split[1] =="wav":
                stimfiles.append(f)
        else:
            if f[:5] == "spike":
                spikefiles.append(f)
    spikefiles = np.sort(spikefiles)
    stimfiles = np.sort(stimfiles)

    stims,durations = load_sound_data(stimfiles,root+model,dsample=t_dsample,sres=nspec,gammatone=gammatone,compress=compress)
    spikes_data, spiky_data = load_spikes_data(spikefiles,root+model, durations,ts_file=False)
    
    return stims,durations,spikes_data,spiky_data

class hyper_opt:
    def __init__(self,stims,psth,filt,nspec,tlen,norm=True,center=True):
        from models import cosstrf
        self.filt=filt
        self.tlen=tlen
        self.nspec=nspec
        self.stims=stims
        self.psth=psth
        self.norm=norm
        self.center=center
        self.model=cosstrf
    
    def run(self,theta):
        c,n,l = np.rint(theta).astype(int)
    
        try: 
            tbas, fromt, tot = cosbasis(self.tlen,n,l,retfn=True,norm=True)
            SPEC,TIM = factorize(self.filt,c)
            filt_start = np.hstack((SPEC.flatten(),tot(TIM).flatten()))
            strf_model = self.model(c,self.nspec,self.tlen,n,l,normalize=self.norm,center=self.center)
            strf_model.set(filt_start)

            out = 0
            for s,p in zip(self.stims,self.psth):
                out -= np.sum(np.power(strf_model.run(s) - p,2))
            
            k = c*(n+self.nspec)
            
            return 2*k-2*out
        except (ValueError,np.linalg.LinAlgError) as e:
            return np.inf
        
    def search(self,bounds=[slice(1,10,1),slice(2,30,1),slice(1,1e3,1e2)]):
        from scipy.optimize import brute
        return np.rint(brute(self.run,bounds)).astype(int)   
