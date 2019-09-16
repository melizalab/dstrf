#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1

CELLFILE="/home/data/crcns/cell_stim_classes.csv"
APARAMS="--save-data --mcmc --skip-completed"

grep "conspecific" $CELLFILE |
    parallel --skip-first-line --colsep ',' python scripts/assimilate.py -k data.cell={1} $APARAMS --restart results/crcns_{1}_xval.npz config/crcns.yml results/crcns_{1}_samples.npz
