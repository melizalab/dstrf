#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1
#export THEANO_FLAGS="base_compiledir=/scratch/dmeliza/.theano"

APARAMS="-k data.trials=50"

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' python scripts/predict_simulated.py $APARAMS config/song_dynamical.yml results/phasic_{1}_samples.npz results/phasic_{1}_predict.npz

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' python scripts/predict_simulated.py $APARAMS config/song_dynamical.yml results/tonic_{1}_samples.npz results/tonic_{1}_predict.npz

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' python scripts/predict_simulated.py $APARAMS config/song_dynamical.yml results/posp_{1}_samples.npz results/posp_{1}_predict.npz
