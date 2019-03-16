# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" This module will load and/or generate stimuli """
from __future__ import print_function, division

import numpy as np


def randn(cf, random_seed=None):
    """Gaussian white noise.

    cf.model.dt
    cf.data.dt
    cf.data.stimulus.duration
    cf.data.filter.nfreq
    (cf.data.random_seed)
    (cf.data.stimulus.intro)
    """
    n_freq = cf.data.filter.get("nfreq", 1)
    n_bins = int(cf.data.stimulus.duration / cf.model.dt)
    upsample = int(cf.data.dt / cf.model.dt)
    n_frames = n_bins // upsample

    np.random.seed(random_seed or cf.data.random_seed)
    stim = np.random.randn(n_freq, n_frames)
    try:
        stim[:, :cf.data.stimulus.intro] = 0
    except AttributeError:
        pass

    return [{"stim": stim, "stim_dt": cf.data.dt, "duration": cf.data.stimulus.duration}]


def crcns(cf):
    """Songs from the CRCNS data set

    cf.data.dt
    cf.data.stimulus.cell
    cf.data.stimulus.root
    cf.data.stimulus.stim_type
    cf.data.stimulus.spectrogram.window
    cf.data.stimulus.spectrogram.f_min
    cf.data.stimulus.spectrogram.f_max
    cf.data.stimulus.spectrogram.compress
    cf.data.stimulus.spectrogram.gammatone
    cf.data.filter.ntau
    (cf.data.filter.nfreq)
    """
    from dstrf import io
    n_freq = cf.data.filter.get("nfreq", 1)

    cspec = cf.data.stimulus.spectrogram

    data = io.load_crcns(cf.data.stimulus.cell,
                         cf.data.stimulus.stim_type,
                         cf.data.stimulus.root, cspec.window,
                         cf.data.dt,
                         f_min=cspec.f_min / 1000,
                         f_max=cspec.f_max / 1000,
                         f_count=n_freq,
                         compress=cspec.compress,
                         gammatone=cspec.gammatone)

    return io.pad_stimuli(data, 0.0, cf.data.filter.ntau * cf.data.dt, fill_value=0.0)