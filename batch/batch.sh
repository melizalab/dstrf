SCRIPT=$1
CELLS=$2
BURN=$3
DIR=$4

while read CELL
do 
	echo $CELL
	/scratch/dstrf/batch/fit.sh $SCRIPT $CELL $BURN $DIR
done < $CELLS 