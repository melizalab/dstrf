#!/bin/sh

CELLS=$1
STIM=$2
DIR=$3

while read CELL
do 
	echo ./crcns-dstrf.sh $CELL $STIM $DIR | batch
done < $CELLS 

