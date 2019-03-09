#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1

scripts/glm_univariate_biocm_emcee.py config/univariate_posp.yml
