#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1
#export THEANO_FLAGS="base_compiledir=/scratch/dmeliza/.theano"

APARAMS="-k data.filter.rf={1} -k data.trials=50"

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' python scripts/predict.py ${APARAMS} -k data.filter.rf={1} -k data.dynamics.model=models/tonic.yml --save-data results/20191013_songtwin/tonic_{1}_pred.npz config/song_dynamical.yml results/20191013_songtwin/tonic_{1}_mcmc.npz

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' python scripts/predict.py ${APARAMS} -k data.filter.rf={1} -k data.dynamics.model=models/phasic.yml --save-data results/20191013_songtwin/phasic_{1}_pred.npz config/song_dynamical.yml results/20191013_songtwin/phasic_{1}_mcmc.npz

