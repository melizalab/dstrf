#!/bin/bash
. /scratch/dstrf/virtualenv/dstrf/bin/activate

echo $1 $2
echo -n $$ START ' ' ; date "+%F %T"

PYSCRIPT=/scratch/dstrf/glm/glmroth.py

DIR=$2/$1/
mkdir $DIR

cp $PYSCRIPT glmroth_$$.py

TAG=$(date '+%F-%H%M')

python2 glmroth_$$.py $1 $DIR $TAG > $DIR/$1_$TAG.log

rm glmroth_$$.py
echo -n $$ DONE ' ' ; date "+%F %T"

deactivate
