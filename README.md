This directory holds code for the dynamical STRF project.

## Setup (virtualenv)

- Create a virtualenv: `python3 -m venv venv && source venv/bin/activate`
- Install dependencies: `pip install -r dev-requirements.txt`
- Configure ipython notebook with this virtualenv as a kernel: `python -m ipykernel install --user --name=dstrf`

## Setup (anaconda):

- Create an environment and populate with

## Running notebooks:

- Start the jupyter notebook server (preferably in a screen): `jupyter notebook` (or `jupyter-3.5 notebook`)

## Analysis workflow

### CRCNS

1. Get duration, average rate, e/o correlation for all cells: `batch/crcns_check.sh`. This generates results/crcns_initial_data.tbl
1. Initial xvalidated MLE fit: `batch/crcns_xval.sh`. This runs fairly quickly on a single host using unconstrained validation. Move the results to a subdirectory under `results`
1. Check initial estimates against parameter bounds: `python scripts/check_param_bounds.py config/crcns.yml results/<subdir>/*.npz > results/crcns_bounds_check.tbl`
1. Check which cells need to be rerun with constraints: In R, `source R/select_for_constrained.R`
1. The constrained fits are much slower and can be parallelized on rivanna. Copy `crcns_needs_constrained.csv` to rivanna, then run `sbatch -p standard -t 100:00:00 --array=1-N batch/crcns_xval_constrained.slurm`, replacing `N` with the number of cells (line count minus 1) in `crcns_needs_constrained.csv`
