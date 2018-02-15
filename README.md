This directory holds code for the dynamical STRF project.

This branch is a fork of the main project that uses the formal likelihood for
the MAT model

## Setup:

- Create a virtualenv: `python3 -m venv ~/.virtualenvs/dstrf && workon dstrf`
- Install dependencies: `pip install -r dev-requirements.txt`
- Configure ipython notebook with this virtualenv as a kernel: `python -m ipykernel install --user --name=dstrf`

## Running notebooks:

- Start the jupyter notebook server (preferably in a screen): `jupyter notebook` (or `jupyter-3.5 notebook`)

## Running jobs on an MPI cluster

This isn't really needed any more.
