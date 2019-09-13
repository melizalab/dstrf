#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=/scratch/dmeliza/.theano"

APARAMS="--save-data --mcmc --skip-completed"

grep -v "^#" config/hg_filters.csv |
    parallel -j 1 --skip-first-line --colsep ',' python scripts/assimilate.py -k data.filter.rf={1} -k data.dynamics.model=models/phasic.yml --restart results/phasic_{1}_xval.npz $APARAMS config/song_dynamical.yml results/phasic_{1}_samples.npz

grep -v "^#" config/hg_filters.csv |
    parallel -j 1 --skip-first-line --colsep ',' python scripts/assimilate.py -k data.filter.rf={1} -k data.dynamics.model=models/tonic.yml --restart results/tonic_{1}_xval.npz $APARAMS config/song_dynamical.yml results/tonic_{1}_samples.npz

grep -v "^#" config/hg_filters.csv |
    parallel -j 1 --skip-first-line --colsep ',' python scripts/assimilate.py -k data.filter.rf={1} -k data.dynamics.model=models/posp.yml -k data.dynamics.current_scaling=2.0 --restart results/posp_{1}_xval.npz $APARAMS config/song_dynamical.yml results/posp_{1}_samples.npz
