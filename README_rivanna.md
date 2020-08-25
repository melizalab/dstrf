
# How to run assimilation tasks on rivanna

## Installation

Clone the dstrf repo, then activate the required modules:

``` shell
module load anaconda/5.2.0-py3.6
module load gcc/7.1.0
module load fftw/3.3.6
module load boost/1.68.0
```

Then create a conda environment for dstrf:

``` shell
conda create -n dstrf
source activate dstrf
conda install theano pandas scikit-learn PyYAML Munch cython
pip install toelis ewave quickspikes
pip install --extra-index-url https://gracula.psyc.virginia.edu/public/pypa/ emcee-tools
```

Several packages have to be installed from source: libtfr, mat-neuron, and pyspike. Clone the repositories and run `python setup.py install` in each.

## Running jobs

Note, for some reason the conda env doesn't activate properly sometimes. The easiest way to fix this seems to be to log out of the frontend and back in again.

### Interactive

``` shell
ijob -c 1 -A meliza -p standard --time 1:00:00
```

Within the allocated shell, run the following to activate the environment:

``` shell
module purge
module load gcc/7.1.0
module load fftw/3.3.6
module load boost/1.68.0
module load anaconda/5.2.0-py3.6
source activate dstrf
```

Test a quick assimilation:

``` shell
python scripts/assimilate.py config/univariate_phasic.yml results/univariate_phasic.npz
```
