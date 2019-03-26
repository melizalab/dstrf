#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=/scratch/dmeliza/.theano"

APARAMS="--save-data --mcmc --skip-completed"

grep -v "^#" config/hg_filters.csv |
    parallel -j 1 --skip-first-line --colsep ',' python scripts/assimilate_simulated.py -k data.filter.rf={1} $APARAMS config/song_phasic.yml results/phasic_{1}.npz

grep -v "^#" config/hg_filters.csv |
    parallel -j 1 --skip-first-line --colsep ',' python scripts/assimilate_simulated.py -k data.filter.rf={1} $APARAMS config/song_tonic.yml results/tonic_{1}.npz

grep -v "^#" config/hg_filters.csv |
    parallel -j 1 --skip-first-line --colsep ',' echo python scripts/assimilate_simulated.py -k data.filter.rf={1} $APARAMS config/song_posp.yml results/posp_{1}.npz
