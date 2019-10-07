#!/bin/bash
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o songtwin_tonic_%a_xval.log
#SBATCH -p standard
#SBATCH -A meliza

module purge
module load gcc/7.1.0
module load fftw/3.3.6
module load boost/1.68.0
module load anaconda/5.2.0-py3.6
source activate dstrf

export OMP_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=/scratch/cdm8j/.theano_${SLURM_ARRAY_TASK_ID}"
CELL=${SLURM_ARRAY_TASK_ID}

python scripts/assimilate.py -k data.root=/scratch/cdm8j/dstrf_sim -k data.filter.rf=${CELL} -k data.dynamics.model=models/tonic.yml --xval ${APARAMS} config/song_dynamical.yml /scratch/cdm8j/dstrf/results/xval/tonic_${CELL}_xval.npz
