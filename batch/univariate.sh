#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1

python scripts/glm_univariate_biocm_emcee.py config/univariate_posp.yml
python scripts/glm_univariate_biocm_emcee.py config/univariate_tonic.yml
python scripts/glm_univariate_biocm_emcee.py config/univariate_phasic.yml
