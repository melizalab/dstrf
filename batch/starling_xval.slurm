#!/bin/bash
#SBATCH -n 1
#SBATCH -t 100:00:00
#SBATCH -o starling_%a_xval.log
#SBATCH -p dev
#SBATCH -A meliza

module purge
module load gcc/7.1.0
module load fftw/3.3.6
module load boost/1.68.0
module load anaconda/5.2.0-py3.6
source activate dstrf

let TMPIDX=${SLURM_ARRAY_TASK_ID}+1000
export OMP_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=/scratch/cdm8j/.theano_${TMPIDX}"
: ${CELLFILE:="/home/cdm8j/dstrf/config/starling_torun.tbl"}
CELL=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}+1) {print \$1}}" < ${CELLFILE})

echo "analyzing ${CELL}"
python scripts/assimilate.py -k data.root=/scratch/cdm8j/ -k data.cell=${CELL} ${PARAMS} config/starling.yml /scratch/cdm8j/dstrf/results/starling/starling_${CELL}_xval.npz
