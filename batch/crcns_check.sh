#!/bin/bash
# checks e/o and spike rate for all data files

. venv/bin/activate
export OMP_NUM_THREADS=1

CELLFILE="/home/data/crcns/cell_stim_classes.csv"
OUTDIR="results"

mkdir -p ${OUTDIR}
grep "conspecific" ${CELLFILE} | cut -d ',' -f1 | xargs python scripts/check_data.py config/crcns.yml > ${OUTDIR}/crcns_initial_data.tbl
