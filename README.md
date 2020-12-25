This repository contains code to run the analyses for Fehrman and Meliza (2021).

The concept behind this project is the "linear-dynamical cascade" model, which combines a linear spectrotemporal RF (STRF) with a detailed dynamical model of neurons in CM, a secondary auditory area in the songbird brain. In this study, we used generalized linear models to examine the effects of a low-threshold potassium current found in some CM neurons on the encoding and adaptation properties of these neurons.

This README will guide you through the process of running an example analysis. In the paper, we examined a large collection of different receptive field types, using the University of Virginia's Rivanna computing cluster to parallelize the analysis. See README_rivanna.md for more details on how to run an analysis in a similar system.

## Setup (docker)

## Setup (virtualenv)

- Create a virtualenv: `python3 -m venv venv && source venv/bin/activate`
- Install dependencies: `pip install -r dev-requirements.txt`
- Configure ipython notebook with this virtualenv as a kernel: `python -m ipykernel install --user --name=dstrf`

## Download stimuli

The zebra finch songs used in our study can be retrieved from https://doi.org/10.6084/m9.figshare.13438109 as a zip file. Unpack the files into a directory called `zf_songs`. Or use your own stimuli!

## Running notebooks:

- Start the jupyter notebook server (preferably in a screen): `jupyter notebook` (or `jupyter-3.5 notebook`). You'll

## Analysis workflow


### Assimilation



### CRCNS

1. Get duration, average rate, e/o correlation for all cells: `batch/crcns_check.sh`. This generates results/crcns_initial_data.tbl
1. Initial xvalidated MLE fit: `batch/crcns_xval.sh`. This runs fairly quickly on a single host using unconstrained validation. Move the results to a subdirectory under `results`
1. Check initial estimates against parameter bounds: `python scripts/check_param_bounds.py config/crcns.yml results/<subdir>/*.npz > results/crcns_bounds_check.tbl`
1. Check which cells need to be rerun with constraints: In R, `source("scripts/select_for_constrained.R")`
1. The constrained fits are much slower and can be parallelized on rivanna. Copy `crcns_needs_constrained.csv` to rivanna, then run `sbatch -p standard -t 100:00:00 --array=1-N batch/crcns_xval_constrained.slurm`, replacing `N` with the number of cells (line count minus 1) in `crcns_needs_constrained.csv`
1. rsync the npz files back to the lab servers, then check again. Note that there may be a number of files that don't fall in the bounds, but these are cells that were excluded by `scripts/select_for_constrained.R`
