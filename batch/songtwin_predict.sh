#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1
#export THEANO_FLAGS="base_compiledir=/scratch/dmeliza/.theano"

APARAMS="-k data.trials=50"

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' echo python scripts/predict_simulated.py $APARAMS config/song_dynamical.yml results/phasic_{1}_samples.npz results/phasic_{1}_predict.npz

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' python scripts/assimilate_simulated.py -k data.filter.rf={1} -k data.dynamics.model=models/tonic.yml --restart results/tonic_{1}_xval.npz $APARAMS config/song_dynamical.yml results/tonic_{1}_samples.npz

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' python scripts/assimilate_simulated.py -k data.filter.rf={1} -k data.dynamics.model=models/posp.yml --restart results/posp_{1}_xval.npz $APARAMS config/song_dynamical.yml results/posp_{1}_samples.npz
