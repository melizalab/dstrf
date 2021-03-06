# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" This module will load stimuli and responses """

from __future__ import print_function, division

import numpy as np


def randomize_stimuli(data, random_seed=10):
    """Randomly shuffle stimuli in a dataset"""
    pass


def randn(cf, random_seed=None):
    """Gaussian white noise. Assumes responses will be simulated.

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

    np.random.seed(random_seed or cf.data.stimulus.random_seed)
    stim = np.random.randn(n_freq, n_frames)
    try:
        stim[:, : cf.data.stimulus.intro] = 0
    except AttributeError:
        pass

    return [
        {"stim": stim, "stim_dt": cf.data.dt, "duration": cf.data.stimulus.duration}
    ]


def wavefiles(cf):
    """Load stimuli from a directory of wave files. Assumes responses will be simulated.

    cf.data.dt
    cf.data.root
    cf.data.stimulus.spectrogram.window
    cf.data.stimulus.spectrogram.f_min
    cf.data.stimulus.spectrogram.f_max
    cf.data.stimulus.spectrogram.f_count
    cf.data.stimulus.spectrogram.compress
    cf.data.stimulus.spectrogram.gammatone
    cf.model.filter.len
    """
    from dstrf import io

    cspec = cf.data.stimulus.spectrogram
    data = io.load_wavefiles(None, cf.data.root, step=cf.data.dt, **cspec)

    io.pad_stimuli(data, 0.0, cf.model.filter.len * cf.data.dt, fill_value=0.0)
    return data


def crcns(cf):
    """Loads responses from the CRCNS data set. Pads the stimuli and preprocesses the spikes.

    cf.data.dt
    cf.data.root
    cf.data.cell
    cf.data.stimulus.stim_type
    cf.data.stimulus.spectrogram.window
    cf.data.stimulus.spectrogram.f_min
    cf.data.stimulus.spectrogram.f_max
    cf.data.stimulus.spectrogram.compress
    cf.data.stimulus.spectrogram.gammatone
    cf.data.stimulus.spectrogram.prepad
    cf.model.filter.len
    cf.model.dt
    cf.model.ataus
    """
    from dstrf import io

    cspec = cf.data.stimulus.spectrogram
    # default option for cell when we just care about the stimuli
    cell = cf.data.get("cell", "blabla0903_2_B")
    data = io.load_crcns(
        cell, cf.data.stimulus.stim_type, cf.data.root, step=cf.data.dt, **cspec
    )

    io.pad_stimuli(
        data, cf.data.prepadding, cf.model.filter.len * cf.data.dt, fill_value=0.0
    )
    io.preprocess_spikes(data, cf.model.dt, cf.model.ataus)
    return data


def neurobank(cf):
    """Loads responses from a neurobank data set. Pads the stimuli and preprocesses the spikes.

    cf.data.dt
    cf.data.root (specify alternate base for archive paths)
    cf.data.cell
    cf.data.stimulus.include (list of stimuli to include in analysis)
    cf.data.stimulus.spectrogram.window
    cf.data.stimulus.spectrogram.f_min
    cf.data.stimulus.spectrogram.f_max
    cf.data.stimulus.spectrogram.compress
    cf.data.stimulus.spectrogram.gammatone
    cf.data.stimulus.spectrogram.prepad
    cf.model.filter.len
    cf.model.dt
    cf.model.ataus
    """
    from dstrf import io

    cspec = cf.data.stimulus.spectrogram
    # default option for cell when we just care about the stimuli
    cell = cf.data.get("cell", "st348_4_6_4")
    data = io.load_neurobank(
        cell,
        step=cf.data.dt,
        stimuli=cf.data.stimulus.get("include", None),
        alt_base=cf.data.get("root", None),
        **cspec
    )
    # A lot cells in this dataset have an unequal number of trials per stimulus.
    # In principle, this would not be an issue, but because of how multiple
    # trials are handled to avoid repeating the stimulus and using up tons of
    # memory, we have to clip the extra trials
    io.clip_trials(data)
    io.pad_stimuli(
        data, cf.data.prepadding, cf.model.filter.len * cf.data.dt, fill_value=0.0
    )
    io.preprocess_spikes(data, cf.model.dt, cf.model.ataus)
    return data


def dstrf_sim(cf):
    """Songs from the dstrf_sim data set

    cf.data.dt
    cf.data.root
    cf.data.stimulus.spectrogram.window
    cf.data.stimulus.spectrogram.f_min
    cf.data.stimulus.spectrogram.f_max
    cf.data.stimulus.spectrogram.f_count
    cf.data.stimulus.spectrogram.compress
    cf.data.stimulus.spectrogram.gammatone
    cf.model.filter.len
    """
    from dstrf import io

    cspec = cf.data.stimulus.spectrogram
    # default option for cell when we just care about the stimuli
    cell = cf.data.get("cell", "b-tonic-24")

    data = io.load_dstrf_sim(cell, cf.data.root, step=cf.data.dt, **cspec)

    io.pad_stimuli(data, 0.0, cf.model.filter.len * cf.data.dt, fill_value=0.0)
    io.preprocess_spikes(data, cf.model.dt, cf.model.ataus)
    return data
