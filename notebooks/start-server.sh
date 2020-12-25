#!/bin/bash
. venv/bin/activate
export OMP_NUM_THREADS=1
jupyter-notebook --no-browser
