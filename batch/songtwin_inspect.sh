#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1
#export THEANO_FLAGS="base_compiledir=/scratch/dmeliza/.theano"

APARAMS="-k data.filter.rf={1}"

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' python scripts/plot_data_singlefig.py config/song_dynamical.yml $APARAMS -k data.dynamics.model=models/phasic.yml results/20191013_songtwin/phasic_{1}_mcmc.npz results/phasic_{1}_predict.npz inspect parameters/Estimated.csv

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' python scripts/plot_data_singlefig.py config/song_dynamical.yml $APARAMS -k data.dynamics.model=models/tonic.yml results/20191013_songtwin/tonic_{1}_mcmc.npz results/tonic_{1}_predict.npz inspect parameters/Estimated.csv



