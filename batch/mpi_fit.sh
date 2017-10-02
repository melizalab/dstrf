#!/bin/bash
#SBATCH -J neurofit_mpi-batch
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --partition=economy
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.log
#SBATCH -N 1
#SBATCH --ntasks-per-node=10

module load python/2.7.6 boost
module load openmpi/intel

source $HOME/dstrf/venv/bin/activate

echo -n $$ START ' ' ; date "+%F %T"

PYSCRIPT=$1
CELL=$2
BURN=$3
SAVEROOT=$4

echo $CELL $BURN $SAVEROOT

DIR=$SAVEROOT/$CELL/
mkdir -p $DIR

TAG=$(date '+%F')-$SLURM_JOB_ID
cp $PYSCRIPT $DIR/$CELL\_$TAG.py

mpiexec --mca btl sm,tcp,self --mca bt1_tcp_if_include eth1 python2 $PYSCRIPT $CELL $BURN $DIR $TAG > $DIR/$CELL\_$TAG.log

echo -n $$ DONE ' ' ; date "+%F %T"
