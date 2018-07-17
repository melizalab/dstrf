from __future__ import print_function, division
import numpy as np
import scipy as sp
import sys
import os
from matplotlib.pyplot import * # plotting functions
import seaborn as sns           # data visualization package
sys.path.append("../") # for importing utils and glm
import pickle
import yaml


def load_mldat(folder,root="/home/tyler/dstrf_results/",exclude=[],EOcut=0.2,CORcut=0.2):

    path = "/".join([root,folder,""])
    results = {}
    
    for root, directories, filenames in os.walk(path):
        for filename in filenames: 
            if root.split("/")[6] in exclude: pass
            else:
                name, ext = filename.split(".")
                if ext == "dat":
                    with open(root+"/"+filename, 'rb') as interfile:
                        try:
                            results[root.split("/")[-1]] = pickle.load(interfile)
                        except:
                            print("Error loading {}".format(name))
                            pass
                    
    maxlik = np.asarray([results[m]["w0"][:3] for m in results.keys() if results[m] and results[m]["eo"]>=EOcut and results[m]["corr"]>=CORcut])
    return results,maxlik


def load_emdat(folder,root="/home/tyler/dstrf_results/",exclude=[],EOcut=0.2,CORcut=0.2):

    path = "/".join([root,folder,""])
    results = {}
    
    for root, directories, filenames in os.walk(path):
        for filename in filenames: 
            if root.split("/")[6] in exclude: pass
            else:
                name, ext = filename.split(".")[-2:]
                if ext == "npz":
                    try:
                        results[root.split("/")[-1]] = np.load(os.path.join(root,filename))
                    except:
                        print("Error loading {}".format(name))
                        
    maxlik = np.asarray([results[m]["w1"][:3] for m in results.keys() if results[m] and results[m]["eo"]>=EOcut and results[m]["corr"]>=CORcut])
    return results,maxlik

def load_results(folder,root="/home/tyler/dstrf_results/",exclude=[],EOcut=0.2,CORcut=0.2):
    results = {}
    res1, ml = load_mldat(folder,root,exclude,EOcut,CORcut)
    res2, mp = load_emdat(folder,root,exclude,EOcut,CORcut)
    for k in res2.keys():
        results[k] = {**res1[k], **res2[k]}
    return results,mp,ml

def crnplt(data,label="",hist=True,mrange=None,labs=[r'$\omega$',r'$\alpha_1$',r'$\alpha_2$']):
    for i in range(3):
        for j in range(3):
            if j > i: break
            subplot(3,3,3*i+j+1)

            if i == j:
                sns.distplot(data[:,j],hist=hist,label=label)
                if mrange:
                    xlim(mrange[0][i])
                if i == 0 and j == 0: legend()
                else: legend().set_visible(False)
            
            else:
                plot(data[:,j],data[:,i],'.',alpha=0.5,markersize=10)
                
                if mrange:
                    xlim(mrange[0][j])
                    ylim(mrange[0][i])
            
                
            xticks(rotation=45)    
            if 3*i+j == 3: ylabel(labs[1]);
            if 3*i+j == 6: ylabel(labs[2])
            if 3*i+j == 0: ylabel(labs[0])
            if 3*i+j == 7: xlabel(labs[1])
            if 3*i+j == 8: xlabel(labs[2])
            if 3*i+j == 6: xlabel(labs[0])
                
    
    sns.despine()
    tight_layout()
    
def make_mlob(cell,yfile="/scratch/tyler/dstrf/scripts/modeldata.yml",krank=2):
    from scipy import signal as ss
    from dstrf import strf, mle, io, performance, spikes
    with open(yfile,"r") as yf:
        config = yaml.load(yf)

    # set variables based on `config`
    ntaus = len(config["mat"]["taus"])
    mat_fixed = np.asanyarray(config["mat"]["taus"] + [config["mat"]["refract"]],dtype='d')
    upsample = int(config["strf"]["stim_dt"] / config["mat"]["model_dt"])
    kcosbas = strf.cosbasis(config["strf"]["ntau"], config["strf"]["ntbas"])
    ntbas = kcosbas.shape[1]
    
    data = io.load_rothman(cell,config["data"]["root"],
                     config["strf"]["spec_window"],
                     config["strf"]["stim_dt"],
                     f_min=config["strf"]["f_min"],
                     f_max=config["strf"]["f_max"], f_count=50,
                     compress=config["strf"]["spec_compress"],
                     gammatone=config["strf"]["gammatone"])
        
    pad_after = config["strf"]["ntau"] * config["strf"]["stim_dt"] # how much to pad after offset
   
    io.pad_stimuli(data, config["data"]["pad_before"], pad_after, fill_value=0.0)
    io.preprocess_spikes(data, config["mat"]["model_dt"], config["mat"]["taus"])
    
    n_test = int(config["data"]["p_test"] * len(data))
    
    # split into assimilation and test sets and merge stimuli
    assim_data = io.merge_data(data[:-n_test])
    test_data = io.merge_data(data[-n_test:])

    assim_data["stim"] = ss.resample(assim_data["stim"],config["strf"]["nfreq"])
    test_data["stim"] = ss.resample(test_data["stim"],config["strf"]["nfreq"])

    n_test = int(config["data"]["p_test"] * len(data))
    
    try:
        mlest = mle.matfact(assim_data["stim"], kcosbas, krank, assim_data["spike_v"], assim_data["spike_h"],
                            assim_data["stim_dt"], assim_data["spike_dt"], nlin=config["mat"]["nlin"])

        mltest = mle.matfact(test_data["stim"], kcosbas, krank, test_data["spike_v"], test_data["spike_h"],
                             test_data["stim_dt"], test_data["spike_dt"], nlin=config["mat"]["nlin"])
    except: 
        mlest = mle.matfact(assim_data["stim"], kcosbas, krank, assim_data["spike_v"], assim_data["spike_h"],
                            assim_data["stim_dt"], assim_data["spike_dt"], nlin=config["mat"]["nlin"])

        mltest = mle.matfact(test_data["stim"], kcosbas, krank, test_data["spike_v"], test_data["spike_h"],
                             test_data["stim_dt"], test_data["spike_dt"], nlin=config["mat"]["nlin"])
        
    return mlest, mltest, assim_data, test_data