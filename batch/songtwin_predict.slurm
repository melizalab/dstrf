#!/bin/bash
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -o songtwin_%a_predict.log
#SBATCH -p standard
#SBATCH -A melizalab

module purge
module load gcc/7.1.0
module load fftw/3.3.6
module load boost/1.68.0
module load anaconda/5.2.0-py3.6
source activate dstrf

export OMP_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=/scratch/cdm8j/.theano_${SLURM_ARRAY_TASK_ID}"
CELL=${SLURM_ARRAY_TASK_ID}
BASE=/scratch/cdm8j/dstrf/results/songtwin

python scripts/predict.py -k data.root=/scratch/cdm8j/dstrf_sim -k data.filter.rf=${CELL} -k data.dynamics.model=models/${MODEL}.yml -k data.trials=50 -p ${BASE}/${MODEL}_${CELL}${SUFFIX}_params.json --save-data ${BASE}/${MODEL}_${CELL}${SUFFIX}_pred.npz config/song_dynamical.yml ${BASE}/${MODEL}_${CELL}${SUFFIX}.npz
