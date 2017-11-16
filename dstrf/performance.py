# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for quantifying model performance"""
from __future__ import print_function, division, absolute_import

import numpy as np
import pyspike as spk


def corrcoef(a, b, downsample=None, smooth=None):
    """Correlation coefficient between PSTHs

    a, b:       spike vectors to compare (nbins, ntrials).
                Must be same number of bins
    downsample: factor by which to downsample the PSTHs
    smooth:     if not None, smooth the downsampled PSTHs

    """
    from dstrf.spikes import psth
    p1 = psth(a, downsample, smooth)
    p2 = psth(b, downsample, smooth)
    return np.corrcoef(p1, p2)[0][1]


def evenoddcorr(spikes, duration, smooth=1, dsample=10):
    """The correlation between PSTHs of even and odd trials"""
    strains = [spk.SpikeTrain(s, duration) for s in spikes]
    evens = strains[::2]
    odds = strains[1::2]
    even_psth = psth_spiky(evens, smooth=smooth, dsample=dsample)
    odd_psth = psth_spiky(odds, smooth=smooth, dsample=dsample)
    return np.corrcoef(even_psth, odd_psth)[0][1]
