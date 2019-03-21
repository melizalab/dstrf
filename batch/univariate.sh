#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=/scratch/dmeliza/.theano"

python scripts/assimilate_simulated.py --mcmc config/univariate_glm.yml results/univariate_glm.npz
python scripts/assimilate_simulated.py --mcmc config/univariate_posp.yml results/biocm_posp_samples.npz
python scripts/assimilate_simulated.py --mcmc config/univariate_tonic.yml results/biocm_tonic_samples.npz
python scripts/assimilate_simulated.py --mcmc config/univariate_phasic.yml results/biocm_phasic_samples.npz
