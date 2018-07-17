SCRIPT=$1
YAML=$2
CELLS=$3
SAVEROOT=$4
MLTAG=$5

. /scratch/tyler/dstrf/venv/bin/activate

mkdir $SAVEROOT
parallel -a $CELLS mkdir $SAVEROOT/{}/

OMP_NUM_THREADS=1 parallel -a $CELLS python3 $SCRIPT {} $YAML $SAVEROOT/{} $MLTAG '>' $SAVEROOT/{}/{}-$MLTAG.log
echo -n $$ DONE ' ' ; date "+%F %T"
exit 0
