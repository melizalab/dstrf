#!/bin/bash

. venv/bin/activate
export OMP_NUM_THREADS=1

INDIR=$1
echo "predicting responses for files in ${INDIR}"

# demangling the file names is potential break point
ls -1 $INDIR/*.npz | sed -nr 's/.*crcns_(\w+)_xval.npz/\1/p' | \
    parallel python scripts/predict.py -k data.cell={} -p ${INDIR}/{}_params.json config/crcns.yml ${INDIR}/crcns_{}_xval.npz

#
