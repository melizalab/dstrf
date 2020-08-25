#!/bin/bash
INDIR=$1
# combine all the json files into a csv file
FIELDS="file.rf,file.glt,w,a1,a2,params_in_bounds,duration,trials_data,rate_mean_data,rate_sd_data,cor_data,trials_pred,rate_mean_pred,rate_sd_pred,cor_pred,binsize"
echo ${FIELDS}
jq -c '. + {file: input_filename | capture("rf(?<rf>[0-9]+)_glt(?<glt>[0-9]+)")}' ${INDIR}/*.json | json2csv -k ${FIELDS}
