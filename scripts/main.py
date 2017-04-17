# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:25:27 2015

@author: Margot
"""

import numpy as np
import matplotlib.pyplot as plt
import spikesim as ss

#Construct the filter
res = 60 #resolution of the filter in pixels
time = 50 #time window of the filter in ms
maxfreq = 8000 #maximum frequency of the signal
latency = 10 #offset between stimulus and response in ms (range: 0 to time)
frequency = 5000 #centering frequency for the filter
A = 0.25 #amplitude of the wavelet peak -- probably don't need to change this
sigma = 0.3 #width of the filter in the time axis -- bigger sigma = narrower time band
gamma = 0.0015 #width of the filter in the frequency axis -- bigger gamma = narrower frequency band
alpha = 1 #depth of inhibitory sidebands on time axis -- bigger alpha = deeper sidebands
beta = 1 #depth of inhibitory sidebands on frequency axis -- bigger beta = deeper sidebands

#ss.strf returns three arguments: h is the strf, t and f are correctly labeled axes for plotting
h, t, f = ss.strf(res,time,maxfreq,latency,frequency,A,sigma,gamma,alpha,beta)

#Plot the strf
#plt.pcolor(t,f,h,vmin=-0.3,vmax=0.3)

#Construct the spectrogram
filename = "../soundstims/modlim/1A556365C919DC0B02CF08D2D5CE4F7B.wav"
#Pxx is the spectrogram, the rest is plotting; plot will show automatically
Pxx = ss.spg(filename,downsample='TRUE',resolution=res)

#Convolve the spectrogram and the filter
r = ss.spgconv(h,Pxx,reshape='FALSE')

#Generate Poisson spike train
spiketrain = ss.spikes(r)

#Generate spike times
#tspike, y = ss.raster(r,1)
#tspike is an array of spike times, y is for plotting

#Save in isk format
#isk = [np.sum(spiketrain[x:x+1600]) for x in np.arange(0,74556,54)]
#np.savetxt('filename.isk',isk)