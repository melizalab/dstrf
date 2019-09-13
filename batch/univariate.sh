#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=/scratch/dmeliza/.theano"

python scripts/assimilate.py --mcmc --save-data config/univariate_glm.yml results/univariate_glm_samples.npz
python scripts/assimilate.py --mcmc --save-data config/univariate_posp.yml results/univariate_posp_samples.npz
python scripts/assimilate.py --mcmc --save-data config/univariate_tonic.yml results/univariate_tonic_samples.npz
python scripts/assimilate.py --mcmc --save-data config/univariate_phasic.yml results/univariate_phasic_samples.npz
