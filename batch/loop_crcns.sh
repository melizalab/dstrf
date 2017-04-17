#!/bin/sh

CELLS=$1
STIM=$2
DIR=$3

while read CELL
do 
    ./crcns-dstrf.sh $CELL $STIM $DIR
done < $CELLS 

