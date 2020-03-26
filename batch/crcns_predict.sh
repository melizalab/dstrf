#!/bin/bash
. venv/bin/activate
export OMP_NUM_THREADS=1

INDIR=$1
#echo "predicting responses for files in ${INDIR}"

# demangling the file names is potential break point
# ls -1 $INDIR/*.npz | sed -nr 's/.*crcns_(\w+)_xval.npz/\1/p' | \
#     parallel python scripts/predict.py -k data.cell={} -p ${INDIR}/{}_params.json config/crcns.yml ${INDIR}/crcns_{}_xval.npz

# combine all the json files into a csv file
FIELDS="cell.name,w,a1,a2,params_in_bounds,duration,trials_data,rate_mean_data,rate_sd_data,cor_data,trials_pred,rate_mean_pred,rate_sd_pred,cor_pred,binsize"
echo ${FIELDS}
jq -c '. + {cell: input_filename | capture("/(?<name>\\w+)_params.json")}' ${INDIR}/*.json | json2csv -k ${FIELDS}
