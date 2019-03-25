#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=/scratch/dmeliza/.theano"

APARAMS="--save-data config/song_dynamical.yml"

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' echo python scripts/assimilate_simulated.py -k data.dynamics.model=models/{1}.yml -k data.filter.rf={2} $APARAMS results/{1}_{2}.npz
