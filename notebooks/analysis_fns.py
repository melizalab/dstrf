from __future__ import print_function, division
import numpy as np
import scipy as sp
import sys
import os
from matplotlib.pyplot import * # plotting functions
import seaborn as sns           # data visualization package
sys.path.append("../") # for importing utils and glm
import pickle


def load_mldat(folder,root="/home/tyler/dstrf_results/",exclude=[],EOcut=0.25,CORcut=0.25):

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
                            results[name] = pickle.load(interfile)
                        except:
                            pass
                    
    maxlik = np.asarray([results[m]["w0"][:3] for m in results.keys() if results[m] and results[m]["eo"]>=EOcut and results[m]["eo"]>=EOcut and results[m]["corr"]>=CORcut])
    return results,maxlik


def load_emdat(folder,root="/home/tyler/dstrf_results/",exclude=[],EOcut=0.25,CORcut=0.25):

    path = "/".join([root,folder,""])
    results = {}
    
    for root, directories, filenames in os.walk(path):
        for filename in filenames: 
            if root.split("/")[6] in exclude: pass
            else:
                name, ext = filename.split(".")
                if ext == "npz":
                    try:
                        results[name] = np.load(os.path.join(root,filename))
                    except:
                        print("Error loading {}".format(name))
                        
    maxlik = np.asarray([results[m]["w1"][:3] for m in results.keys() if results[m] and results[m]["eo"]>=EOcut and results[m]["eo"]>=EOcut and results[m]["corr"]>=CORcut])
    return results,maxlik



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
            if 3*i+j == 0: ylabel(r'$\omega$')
            if 3*i+j == 7: xlabel(labs[1])
            if 3*i+j == 8: xlabel(labs[2])
            if 3*i+j == 6: xlabel(labs[0])
    sns.despine()
    tight_layout()