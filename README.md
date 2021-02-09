This repository contains code to run the analyses for Fehrman et al (2021).

The concept behind this project is the "linear-dynamical cascade" model, which combines a linear spectrotemporal RF (STRF) with a detailed dynamical model of neurons in CM, a secondary auditory area in the songbird brain. In this study, we used generalized linear models to examine the effects of a low-threshold potassium current found in some CM neurons on the encoding and adaptation properties of these neurons.

This README will guide you through the process of running an example analysis. In the paper, we examined a large collection of different receptive field types, using the University of Virginia's Rivanna computing cluster to parallelize the analysis. See README_rivanna.md for more details on how to run an analysis in a similar system.

## Setup with Docker (recommended)

To start the server for the demo notebooks: `docker run -p 8888:8888 -it --rm dmeliza/dstrf:latest`. Copy and paste the URL that appears after the docker container has started.

To start a shell for running scripts: `docker run -it --rm dmeliza/dstrf:latest /bin/bash`

## Setup (virtualenv)

- Create a virtualenv: `python3 -m venv venv && source venv/bin/activate`
- Install dependencies: `pip install -r dev-requirements.txt`

## Download stimuli

The zebra finch songs used in our study can be retrieved from https://doi.org/10.6084/m9.figshare.13438109 as a zip file. Unpack the files into a directory called `zf_songs`. Or use your own stimuli! They just need to be in wave format.

## Example analysis:

To run the demo notebooks, run `notebooks/start-server.sh`. This will create a jupyter notebook server. Navigate your browser to http://localhost:9001 to connect. The notebooks under `notebooks/demos` will demonstrate simulation, GLM estimation, and prediction using several different base models.

To run a complete analysis for one of the RF/dynamics combinations in the paper, run the following code blocks. Change `CELL` and `MODEL` shell variables to pick a different combination.

First, perform maximum likelihood estimation using cross-validation to determine the optimal regularization hyperparameters.

``` shell
export OMP_NUM_THREADS=1
export CELL=0
export MODEL=tonic
python scripts/assimilate.py -k data.filter.rf=${CELL} -k data.dynamics.model=models/${MODEL}.yml --xval config/song_dynamical.yml ${MODEL}_${CELL}_xval.npz
```

Next, use Markov-chain Monte Carlo to sample the posterior around the maximum likelihood estimate. You may want to edit `config/song_dynamical.yml` to adjust the number of threads to match the number of available CPU cores. This step is time-consuming and can be skipped if you don't care about the posterior distribution.

``` shell
python scripts/assimilate.py -k data.filter.rf=${CELL} -k data.dynamics.model=models/${MODEL}.yml --mcmc --restart ${MODEL}_${CELL}_xval.npz config/song_dynamical.yml ${MODEL}_${CELL}_mcmc.npz
```

Finally, generate (posterior) predictive distributions. If you skip the previous step, replace `mcmc` in the command below with `xval`.

``` shell
python scripts/predict.py -k data.filter.rf=${CELL} -k data.dynamics.model=models/${MODEL}.yml -k data.trials=50 -p ${MODEL}_${CELL}_params.json --save-data ${MODEL}_${CELL}_pred.npz config/song_dynamical.yml ${MODEL}_${CELL}_mcmc.npz
```
