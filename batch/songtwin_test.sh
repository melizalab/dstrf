#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1

python scripts/assimilate.py -k data.filter.rf=10 -k data.dynamics.model=models/phasic.yml -k emcee.nwalkers=1000 -k emcee.nsteps=50000 -k emcee.startpos_scale=0.001 --restart results/20190402/phasic_10_xval.npz --save-data --mcmc --save-chain config/song_dynamical.yml phasic_10_samples.npz
