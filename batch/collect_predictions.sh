#!/bin/bash
INDIR=$1
# combine all the json files into a csv file
FIELDS="cell.name,w,a1,a2,params_in_bounds,duration,trials_data,rate_mean_data,rate_sd_data,cor_data,trials_pred,rate_mean_pred,rate_sd_pred,cor_pred,binsize"
echo ${FIELDS}
jq -c '. + {cell: input_filename | capture("/(?<name>\\w+)_params.json")}' ${INDIR}/*.json | json2csv -k ${FIELDS}
