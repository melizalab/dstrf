#!/bin/sh

CELLS=$1
DIR=$2

while read CELL
do 
	echo $CELL
	/scratch/dstrf/batch/matroth.sh $CELL $DIR
done < $CELLS 

