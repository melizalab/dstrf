#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1
APARAMS="--mcmc --save-data --save-chain"
DATE=`date +%Y%m%d`
RESDIR="results/${DATE}_univariate"
mkdir -p ${RESDIR}
echo "univariate analysis - output to ${RESDIR}"

# python scripts/assimilate.py ${APARAMS} config/univariate_glm.yml results/univariate_glm_samples.npz
python scripts/assimilate.py ${APARAMS} config/univariate_tonic.yml ${RESDIR}/univariate_tonic_samples.npz
python scripts/assimilate.py ${APARAMS} config/univariate_phasic.yml ${RESDIR}/univariate_phasic_samples.npz

# python scripts/predict.py config/univariate_glm.yml results/univariate_glm_samples.npz results/univariate_glm_predict.npz
python scripts/predict.py --save-data ${RESDIR}/univariate_tonic_predict.npz config/univariate_tonic.yml ${RESDIR}/univariate_tonic_samples.npz
python scripts/predict.py --save-data ${RESDIR}/univariate_phasic_predict.npz config/univariate_phasic.yml ${RESDIR}/univariate_phasic_samples.npz
