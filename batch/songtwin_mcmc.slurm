#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -t 04:00:00
#SBATCH -o songtwin_%a_mcmc.log
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

python scripts/assimilate.py -k data.root=/scratch/cdm8j/dstrf_sim -k data.filter.rf=${CELL} -k data.dynamics.model=models/${DYNAMICS}.yml --mcmc ${APARAMS} --restart /scratch/cdm8j/dstrf/results/xval/${DYNAMICS}_${CELL}_xval.npz config/song_dynamical.yml /scratch/cdm8j/dstrf/results/mcmc/${DYNAMICS}_${CELL}_mcmc.npz
