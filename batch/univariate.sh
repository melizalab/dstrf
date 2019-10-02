#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1
APARAMS="--mcmc --save-data --save-chain"

python scripts/assimilate.py ${APARAMS} config/univariate_glm.yml results/univariate_glm_samples.npz
python scripts/assimilate.py ${APARAMS} --constrained -k data.dynamics.model=models/posp.yml -k data.dynamics.current_scaling=2.0 config/univariate_dynamical.yml results/univariate_posp_samples.npz
python scripts/assimilate.py ${APARAMS} -k data.dynamics.model=models/tonic.yml -k data.dynamics.current_scaling=4.0 config/univariate_dynamical.yml results/univariate_tonic_samples.npz
python scripts/assimilate.py ${APARAMS} -k data.dynamics.model=models/phasic.yml -k data.dynamics.current_scaling=9.0 config/univariate_dynamical.yml results/univariate_phasic_samples.npz
