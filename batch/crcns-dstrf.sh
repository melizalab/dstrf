#!/bin/sh

echo $1 $2
echo -n $$ START ' ' ; date "+%F %T"

PYSCRIPT=/scratch/tyler/dstrf/dstrf_param_mat.py

DIR=$3/$1/
mkdir $DIR

cp $PYSCRIPT dstrf_crcns_$$.py

TAG=$(date '+%F-%H%M')

python2 dstrf_crcns_$$.py $1 $2 $DIR $TAG> $DIR/$2_$TAG.log

rm dstrf_crcns_$$.py
echo -n $$ DONE ' ' ; date "+%F %T"
