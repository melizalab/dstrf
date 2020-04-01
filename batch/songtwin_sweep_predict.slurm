#!/bin/bash
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -o songtwin_sweep_%a_predict.log
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
GLT=${SLURM_ARRAY_TASK_ID}

python scripts/predict.py -k data.root=/scratch/cdm8j/dstrf_sim -k data.filter.rf=${CELL} -k data.dynamics.model=models/phasic.yml -k data.dynamics.param.g_klt=${GLT} -k data.trials=50 -p /scratch/cdm8j/dstrf/results/xval/sweep_rf${CELL}_glt${GLT}_params.json config/song_dynamical.yml /scratch/cdm8j/dstrf/results/xval/sweep_rf${CELL}_glt${GLT}_xval.npz