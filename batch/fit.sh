#!/bin/bash
. /scratch/dstrf/virtualenv/dstrf/bin/activate

echo -n $$ START ' ' ; date "+%F %T"

PYSCRIPT=$1
CELL=$2
BURN=$3
SAVEROOT=$4

echo $CELL $BURN $SAVEROOT

DIR=$SAVEROOT/$CELL/
mkdir $SAVEROOT
mkdir $DIR

TAG=$(date '+%F-%H%M')
cp $PYSCRIPT $DIR/$CELL\_$TAG.py

python2 $DIR/$CELL\_$TAG.py $CELL $BURN $DIR $TAG > $DIR/$CELL\_$TAG.log

echo -n $$ DONE ' ' ; date "+%F %T"