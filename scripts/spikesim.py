# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:46:13 2015

@author: Margot
"""

#Construct the filter
def strf(resolution=50,time=50,maxfreq=8000,latency=0,frequency=0,A=0.25,sigma=0.1,gamma=0.001,alpha=1.4,beta=1.5):
    import numpy as np    
    scale = resolution/50.0   
    t = np.arange(float(np.negative(time))*0.5,1)
    tscale = np.arange(np.negative(time),1,2)
    x = latency*0.5
    f = np.arange(0,maxfreq+1,float(maxfreq)/resolution)
    y = frequency
    tc = t+x
    fc = f-y
    tprime, fprime = np.meshgrid(tc,fc)
    sigma = sigma/scale
    Gtf = A*np.exp(-sigma**2*tprime**2-gamma**2*fprime**2)*(1-alpha**2*sigma**2*tprime**2)*(1-beta**2*gamma**2*fprime**2)
    return (Gtf,tscale,f)

#Construct the spectrogram
def spg(stim,NFFT=256,noverlap=192,downsample='FALSE',resolution=10): 
    from scipy.io.wavfile import read 
    import matplotlib.pyplot as plt
    import numpy as np 
    stimframes = read(stim)
    Fs = stimframes[0]
    Pxx, freqs, bins, im = plt.specgram(stimframes[1],NFFT=NFFT,Fs=Fs,noverlap=noverlap)
    Pxx = Pxx[:65,:]
    freqs = freqs[:65]
    Pxx = np.concatenate((np.ones((65,12)),Pxx),1)
    Pxx = np.log10(Pxx)
    if downsample == 'TRUE':
        [Ls,Ts]=Pxx.shape
        nf=resolution
        df=freqs[-1]/nf
        f0=freqs[0]+df/2.
        lPxi=np.zeros((nf,Ts));
        fi=np.arange(f0,freqs[-1],df)
        for i in range(0,Ts):
            lPxi[:,i]=np.interp(fi,freqs,Pxx[:,i])
        Pxx=lPxi
    xextent = 0, np.amax(bins)
    xmin, xmax = xextent
    extent = xmin, xmax, freqs[0], freqs[-1]
    plt.figure(1)
    ax = plt.gca()
    imgplot=plt.imshow(np.flipud(10.*Pxx),aspect='auto',cmap=None,extent=extent)
    imgplot.set_clim(-60.,40.)
    plt.colorbar()
    return (Pxx)
    
#def spgconv(h,s):
#    import numpy as np  
#    specbins = len(s[:,0])
#    filterbins = len(h)
#    if specbins%filterbins == 0:
#        reshape = specbins/filterbins
#    else:
#        reshape = (specbins/filterbins)+1
#    stretch = np.zeros((specbins,len(h)))
#    for i in range(specbins):
#        stretch[i] = h[i/reshape]
#    h = stretch[:specbins]
#    s = np.log(s)
#    cstep = 0
#    conv = []    
#    for i in range(len(s[0])):
#        for j in range(filterbins):
#            cstep = cstep + np.dot(s[:,i],h[:,j])
#        conv.append(cstep)
#        cstep = 0
#    return conv
    
#Convolve spectrogram and filter
def spgconv(h,s,reshape='TRUE'):
    import numpy as np
    if reshape == 'TRUE':
        specbins = len(s[:,0])
        filterbins = len(h)
        if specbins%filterbins == 0:
            reshape = specbins/filterbins
        else:
            reshape = (specbins/filterbins)+1
        stretch = np.zeros((specbins,len(h)))
        for i in range(specbins):
            stretch[i] = h[i/reshape]
        h = stretch[:specbins]
    cs = 0
    for i in range(len(s[:,0])):
        conv = np.convolve(s[i,:],h[i,:],'same')
        if type(cs) == int:
            cs = conv
        else:
            cs = np.add(cs,conv)
    cs = (cs/(1+np.abs(cs)))
    return cs 
    
#Simulate spikes and return as 0/1s
def spikes(r):
    import numpy as np  
    for i in range(len(r)):
        if r[i] < 0:
            r[i] = 0
    rP = np.random.binomial(1,r)
    return rP
    
#Simulate spikes and return as spike times
def raster(r,i):
    import numpy as np 
    rP = spikes(r)
    T = np.arange(0,len(rP))
    tspikes = T[np.nonzero(rP)]
    y = np.ones(len(tspikes))
    y = y*i
    return (tspikes,y)
    
#Simulate and plot multiple trials
def rasterplot(r,trials):
    import numpy as np  
    import matplotlib.pyplot as plt
    x,y = raster(r,1)
    n = 100.0/32000
    if trials > 1:
        for i in range(2,trials+1):
            a,b = raster(r,i)
            x = np.concatenate((x,a),1)
            y = np.concatenate((y,b),1)
    plt.cla()
    plt.plot(x*n,y,linestyle='',marker='|')