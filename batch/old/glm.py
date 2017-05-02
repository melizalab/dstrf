import numpy as np
import utils

class cosstrf():
    def __init__(self, channels, nspec, tlen, ncos=10,coslin=1):
        self.channels = channels
        self.nspec = nspec
        self.tlen = tlen
        self.ncos = ncos
        self.cosbas = utils.cosbasis(tlen,ncos,lin=coslin)
        self.invbas = np.linalg.pinv(self.cosbas)
        self.tfilt = None
        self.sfilt = None
        self.filt = None
        
    def set(self,theta):
        flat_sfilt = theta[:self.nspec*self.channels]
        flat_tfilt = theta[self.nspec*self.channels:]
        self.sfilt = np.reshape(flat_sfilt,(self.channels,self.nspec))
        self.tfilt = np.matmul(self.cosbas,np.reshape(flat_tfilt,(self.channels,self.ncos)).T).T
        self.filt  = np.matmul(self.sfilt.T,self.tfilt)

    def run(self,stim):
        return utils.spgconv(self.filt,stim)

class cosstrf_design():
    def __init__(self, channels, nspec, tlen, ncos=10,coslin=1):
        self.channels = channels
        self.nspec = nspec
        self.tlen = tlen
        self.ncos = ncos
        self.cosbas = utils.cosbasis(tlen,ncos,lin=coslin)
        self.invbas = np.linalg.pinv(self.cosbas)
        self.tfilt = None
        self.sfilt = None
        self.filt = None
        
    def set(self,theta):
        flat_sfilt = theta[:self.nspec*self.channels]
        flat_tfilt = theta[self.nspec*self.channels:]
        self.sfilt = np.reshape(flat_sfilt,(self.channels,self.nspec))
        self.tfilt = np.matmul(self.cosbas,np.reshape(flat_tfilt,(self.channels,self.ncos)).T).T
        self.filt  = np.matmul(self.sfilt.T,self.tfilt).flatten()

    def run(self,stim):
        return np.matmul(stim,self.filt)

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
    
class GLM_cos():
    def __init__(self,channels,nspec,tlen,hlen,tcos=10,hcos=8,tcoslin=1,hcoslin=10,nonlin=np.exp,spike=False,dt=0.001):
        self.nonlin = nonlin if nonlin is not None else lambda x: x
        self.k = cosstrf(channels,nspec,tlen,tcos,tcoslin)
        self.h = cosstrf(1,1,hlen,hcos,hcoslin)
        self.post_spike = None
        self.offset = 0
        self.spike=spike
        self.nh = hlen
        self.dt=dt

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
                    times = np.asarray(ob)
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
