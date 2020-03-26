#!/bin/bash
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -o crcns_%a_predict.log
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
CELLFILE="/scratch/cdm8j/crcns/cell_stim_classes.csv"
CELL=$(grep conspecific ${CELLFILE} | awk "{if (NR==${SLURM_ARRAY_TASK_ID}+1) {print \$1}}" | cut -d "," -f 1)

python scripts/predict.py -k data.cell=${CELL} -p /scratch/cdm8j/dstrf/results/params/{}_params.json config/crcns.yml /scratch/cdm8j/dstrf/results/xval/crcns_${CELL}_xval.npz
