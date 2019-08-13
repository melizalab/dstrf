#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=/scratch/dmeliza/.theano"

APARAMS="--xval"

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' python scripts/assimilate_simulated.py -k data.filter.rf={1} -k data.dynamics.model=models/phasic.yml $APARAMS config/song_dynamical.yml results/phasic_{1}_xval.npz

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' python scripts/assimilate_simulated.py -k data.filter.rf={1} -k data.dynamics.model=models/tonic.yml $APARAMS config/song_dynamical.yml results/tonic_{1}_xval.npz

grep -v "^#" config/hg_filters.csv |
    parallel --skip-first-line --colsep ',' python scripts/assimilate_simulated.py -k data.filter.rf={1} -k data.dynamics.model=models/posp.yml -k data.dynamics.current_scaling=2.0 $APARAMS config/song_dynamical.yml results/posp_{1}_xval.npz
