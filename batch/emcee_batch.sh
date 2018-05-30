CELLS=$1
SAVEROOT=$2
MLTAG=$3
EMTAG=$4

while read CELL
do 
	. /scratch/tyler/dstrf/venv/bin/activate
	echo $CELL
	echo -n $$ START ' ' ; date "+%F %T"

	DIR=$SAVEROOT/$CELL/

	OMP_NUM_THREADS=1 krenew -a -K 10 python3 scripts/split_posp-emcee.py $DIR/$CELL'_'$MLTAG-ml.dat $DIR $EMTAG > $DIR/$CELL'_'$MLTAG.log

	echo -n $$ DONE ' ' ; date "+%F %T"
done < $CELLS 
