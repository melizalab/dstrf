#!/bin/bash
. /scratch/dstrf/virtualenv/dstrf/bin/activate

echo $1 $2
echo -n $$ START ' ' ; date "+%F %T"

PYSCRIPT=/scratch/dstrf/mat/matroth.py

DIR=$2/$1/
mkdir $DIR

cp $PYSCRIPT matroth_$$.py

TAG=$(date '+%F-%H%M')

python2 matroth_$$.py $1 $DIR $TAG > $DIR/$1_$TAG.log

rm matroth_$$.py
echo -n $$ DONE ' ' ; date "+%F %T"

deactivate