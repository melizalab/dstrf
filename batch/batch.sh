PYSCRIPT=$1
CELLS=$2
YAML=$3
SAVEROOT=$4


mkdir $SAVEROOT
while read CELL
do 
    
    . /scratch/tyler/dstrf/venv/bin/activate
	echo $CELL
	echo -n $$ START ' ' ; date "+%F %T"

	DIR=$SAVEROOT/$CELL/
	
	mkdir $DIR

	TAG=$(date '+%F-%H%M')
	cp $PYSCRIPT $DIR/$CELL\_$TAG.py

	OMP_NUM_THREADS=1 python3 $DIR/$CELL\_$TAG.py $CELL $YAML $DIR $TAG > $DIR/$CELL\_$TAG.log

	echo -n $$ DONE ' ' ; date "+%F %T"
done < $CELLS 
