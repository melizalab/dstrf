# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for manipulating spike trains"""
from __future__ import print_function, division, absolute_import

import numpy as np
import pyspike as spk


def psth(spike_v, downsample=None, smooth=None):
    """Compute psth from multi-trial spike vector (dimension nbins x ntrials)

    downsample: if not None, factor by which to downsample the PSTH
    smooth:     if not None, smooth the downsampled PSTH
    """
    from scipy.ndimage.filters import gaussian_filter1d

    nbins, ntrials = spike_v.shape
    if downsample is not None:
        psth = np.sum(spike_v.reshape(nbins // downsample, ntrials, -1), axis=(1, 2))
    else:
        psth = np.sum(spike_v, axis=1)
    if smooth is not None:
        return gaussian_filter1d(psth.astype('d'), smooth, mode="constant", cval=0.0)
    else:
        return psth
