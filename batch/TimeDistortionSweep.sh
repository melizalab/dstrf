#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1
#export THEANO_FLAGS="base_compiledir=/scratch/dmeliza/.theano"


parallel python scripts/TimePeakDistortion.py -k data.filter.rf=4 config/song_dynamical.yml {} Figures ::: sweep_results/20200325_songtwin_sweep/sweep_rf4_glt*_xval.npz

parallel python scripts/TimePeakDistortion.py -k data.filter.rf=22 config/song_dynamical.yml {} Figures ::: sweep_results/20200325_songtwin_sweep/sweep_rf22_glt*_xval.npz

parallel python scripts/TimePeakDistortion.py -k data.filter.rf=23 config/song_dynamical.yml {} Figures ::: sweep_results/20200325_songtwin_sweep/sweep_rf23_glt*_xval.npz