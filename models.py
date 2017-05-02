import numpy as np
import utils
import cneurons as cn
import neurofit.utils as nfutils
from scipy.signal import resample


# strf model with time in cosine basis (a la Pillow)
class cosstrf():
    def __init__(self, channels, nspec, tlen, ncos=10,tcoslin=1,normalize=False,center=False):
        self.channels = channels
        self.nspec = nspec
        self.tlen = tlen
        self.ncos = ncos
        self.cosbas = utils.cosbasis(tlen,ncos,lin=tcoslin)
        self.invbas = np.linalg.pinv(self.cosbas)
        self.tfilt = None
        self.sfilt = None
        self.filt = None
        self.normalize = normalize
        self.center = center
        
    def dim(self):
        return self.channels*(self.nspec+self.ncos)

    def set(self,theta):
        flat_sfilt = theta[:self.nspec*self.channels]
        flat_tfilt = theta[self.nspec*self.channels:]
        self.sfilt = np.reshape(flat_sfilt,(self.channels,self.nspec))
        self.tfilt = np.matmul(self.cosbas,np.reshape(flat_tfilt,(self.channels,self.ncos)).T).T
        self.filt  = np.matmul(self.sfilt.T,self.tfilt)

    def run(self,stim):
        r = utils.spgconv(self.filt,stim)
        if self.normalize: r = nfutils.normalize(r,center=self.center)
        return r

# strf model with time in cosine basis (a la Pillow)
class cosstrf_design():
    def __init__(self, channels, nspec, tlen, ncos=10,coslin=1,normalize=False,center=False):
        self.channels = channels
        self.nspec = nspec
        self.tlen = tlen
        self.ncos = ncos
        self.cosbas = utils.cosbasis(tlen,ncos,lin=coslin)
        self.invbas = np.linalg.pinv(self.cosbas)
        self.tfilt = None
        self.sfilt = None
        self.flat = None
        self.filt = None
        self.norm = normalize
        self.center = center
        
    def set(self,theta):
        flat_sfilt = theta[:self.nspec*self.channels]
        flat_tfilt = theta[self.nspec*self.channels:]
        self.sfilt = np.reshape(flat_sfilt,(self.channels,self.nspec))
        self.tfilt = np.matmul(self.cosbas,np.reshape(flat_tfilt,(self.channels,self.ncos)).T).T
        self.filt  = np.matmul(self.sfilt.T,self.tfilt)
        self.flat = np.matmul(self.sfilt.T,self.tfilt).flatten()

    def run(self,stim):
        R = np.matmul(stim,self.flat)
        if self.norm: nfutils.normalize(R,center=self.center)
        return R
    

class LNP_cos():
    def __init__(self,channels,nspec,tlen,ncos=10,coslin=1,nonlin=np.exp,bias=True,spike=False):
        self.nonlin = nonlin
        self.pstrf = cosstrf(channels,nspec,tlen,ncos,coslin)
        self.post_spike = None
        self.bias = bias
        self.offset = 0
        self.spike = spike

    def set(self,theta):
        if self.bias:
            self.offset,theta = theta[0],theta[1:]
        self.pstrf.set(theta)
        
    def run(self,data):
        stim, observed_spikes = data if np.shape(data) == (2,) else (data,None)
            
        synap = self.pstrf.run(stim)
    
        rate = synap + self.offset
        if self.nonlin is not None: rate = self.nonlin(rate)
    
        return rate
    
class LNP_coslin():
    def __init__(self,channels,nspec,tlen,ncos=10,coslin=1,nonlin=np.exp,bias=True):
        self.nonlin = nonlin
        self.post_spike = None
        self.bias = bias
        self.offset = 0
        self.channels=channels
        self.nspec=nspec
        self.tlen=tlen
        self.ncos=ncos

    def set(self,theta):
        if self.bias:
            coslin,self.offset,theta = theta[0],theta[1],theta[2:]
        self.pstrf = cosstrf(self.channels,self.nspec,self.tlen,self.ncos,np.power(2,coslin))
        self.pstrf.set(theta)
        
    def run(self,data):
        stim, observed_spikes = data if np.shape(data) == (2,) else (data,None)
            
        synap = self.pstrf.run(stim)
    
        rate = synap + self.offset
        if self.nonlin is not None: rate = self.nonlin(rate)
    
        return rate
    
# glm model with a cosine basis
class GLM_cos():
    def __init__(self,channels,nspec,tlen,hlen,tcos=10,hcos=8,tcoslin=1,htcoslin=10,nonlin=np.exp,spike=False,dt=0.001):
        self.nonlin = nonlin if nonlin is not None else lambda x: x
        self.k = cosstrf(channels,nspec,tlen,tcos,tcoslin)
        self.h = cosstrf(1,1,hlen,hcos,htcoslin)
        self.post_spike = None
        self.offset = 0
        self.spike=spike
        self.nh = hlen
        self.dt=dt
        self.nspec = nspec
        self.tcos = tcos
        self.hcos = hcos
        self.channels = channels
        
    def dim(self):
        return self.channels*(self.nspec + self.tcos + self.hcos) + 1

    def set(self,theta):
        self.offset,ktheta,htheta = np.split(theta,[1,-self.h.ncos])
        self.k.set(ktheta)
        self.h.set(np.hstack((1,htheta)))
        
    def run(self,data):
        stim, observed_spikes = data if np.shape(data) == (2,) else (data,None)
            
        synap = self.k.run(stim)
        
        duration = synap.size
        
        spikes = np.zeros(duration+1) 
        spike_times = []    
        
        
        if observed_spikes is not None:
            rate = []
            oneobv = False
            if np.shape(observed_spikes[0]) == ():
                observed_spikes = [observed_spikes]
                oneobv = True
            for ob in observed_spikes:    
                if len(ob) >= duration:
                    spikes = ob[:duration]
                    spike_times = np.where(spikes>0)

                else:
                    spikes *= 0
                    times = np.asarray(np.clip(np.rint(ob),0,duration-1),dtype=int)
                    spikes[times+1] = 1
                    spike_times.append(times)

                post = self.h.run([spikes[:duration]])
                rate.append(self.nonlin(synap + post + self.offset)*self.dt)
            if oneobv:
                rate = rate[0]
                spike_times = spike_times[0]

        else:
            rate = np.zeros(duration)
            post_spike = np.fliplr(self.h.filt)
            for i in range(1,duration):
                window = spikes[max(0,i-self.nh):i]
                if i < self.nh: window = np.pad(window,(self.nh-i,0),"constant")
                post = np.dot(post_spike,window)
                r = synap[i] + post + self.offset
                rate[i] = self.nonlin(r)*self.dt
                if np.random.poisson(rate[i]) > 0:
                    spikes[i] = 1
                    spike_times.append(i)   

        return (rate, spike_times) if self.spike else rate

# augmented mat model with R and tm fixed to 1
class mat():
    def __init__(self, free_ts=False):
        self.nrn = None
        self.free_ts = free_ts

    def dim(self):
        return 6 if self.free_ts else 4

    def set(self, theta):
        a, b, c, w = theta[:4]
        self.nrn = cn.augmat(a, b, c, w)
        self.nrn.R = 1
        self.nrn.tm = 1

        if self.free_ts:
            t1, t2 = theta[4:]
            self.nrn.t1 = t1
            self.nrn.t2 = t2

    def run(self, iapp):
        self.nrn.apply_current(iapp, 1)
        return self.nrn.simulate(len(iapp), 1)

# combining the strf and mat models
class dstrf_mat():
    def __init__(self,channels=1,nspec=15,tlen=30,ncos=10,coslin=1,upsample=1,
                 scale=1,free_ts=False,normalize=False,center=False,noise=None):
        self.mat = mat(free_ts=free_ts)
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
        return self.channels*(self.nspec+self.ncos) + ( 6 if self.free_ts else 4 )
     
    def set(self, theta):
        cut = -6 if self.free_ts else -4
        self.pstrf.set(theta[:cut])
        self.mat.set(theta[cut:])
        self.mat.nrn.R = 1
        self.mat.nrn.tm = 1
        
    def run(self, stim):
        r = self.pstrf.run(stim)
        if self.noise is not None: r += np.random.randn(len(r))*self.noise
        r = resample(r,len(r)*self.upsample)*self.scale
        return self.mat.run(r)