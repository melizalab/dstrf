# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for file IO"""
from __future__ import print_function, division, absolute_import

import os
import glob
import toelis as tl
import numpy as np


def load_crcns(cell, stim_type, root, window, step, **specargs):
    """Load stimulus and response data from CRCNS repository

    Additional keyword arguments are passed to load_stimulus()
    """
    spikesroot = os.path.join(root, "all_cells", cell, stim_type)
    stimroot = os.path.join(root, "all_stims")

    out = []
    for fname in glob.iglob(os.path.join(spikesroot, "*.toe_lis")):
        base, ext = os.path.splitext(os.path.basename(fname))
        spec, dur = load_stimulus(os.path.join(stimroot, base + ".wav"), window, step, **specargs)
        spikes = tl.read(open(fname, "rt"))[0]
        out.append({"cell_name": cell,
                    "stim_name": base,
                    "duration": dur,
                    "stim": spec,
                    "stim_dt": step,
                    "spikes": spikes})
    return out


def load_stimulus(path, window, step, f_min=0.5, f_max=8.0, f_count=30,
                  compress=1, gammatone=False):
    """Load sound stimulus and calculate spectrotemporal representation.

    Parameters:

    path: location of wave file
    window: duration of window (in ms)
    step: window step (in ms)
    f_min: minimum frequency (in kHz)
    f_max: maximum frequency (in kHz)
    f_count: number of frequency bins
    gammatone: if True, use gammatone filterbank

    Returns spectrogram, duration (ms)
    """
    import ewave
    fp = ewave.open(path, "r")
    Fs = fp.sampling_rate / 1000.
    osc = fp.read()
    if gammatone:
        import gammatone.gtgram as gg
        Pxx = gg.gtgram(osc, Fs * 1000, window / 1000, step / 1000, f_count, f_min * 1000, f_max * 1000)
    else:
        import libtfr
        # nfft based on desired number of channels btw f_min and f_max
        nfft = int(f_count / (f_max - f_min) * Fs)
        npoints = int(Fs * window)
        if nfft < npoints:
            raise ValueError("window size {} ({} points) too small "
                             "for desired freq resolution".format(window, npoints))

        nstep = int(Fs * step)
        taper = np.hanning(npoints)
        mfft = libtfr.mfft_precalc(nfft, taper)
        Pxx = mfft.mtspec(osc, nstep)
        freqs, ind = libtfr.fgrid(Fs, nfft, [f_min, f_max])
        Pxx = Pxx[ind, :]
    if compress is not None:
        Pxx = np.log10(Pxx + compress)
    return Pxx, Pxx.shape[1] * step


def merge_data(seq, pad_before, pad_after, dt, fill_value=None):
    """Merge a sequence of stimuli into a single trial

    seq:     a sequence of dicts containing {stim, duration, spikes}
    pad_before: the duration of silence (units of dt) to insert before each stimulus
    pad_after: the duration of silence (units of dt) to insert after each stimulus
    dt:      the duration of each frame in the stimulus
    fill_value: sets the value of the padding. If None, this is equal to the
    mean of the first frame of the stimulus.

    If the input sequence contains spikes, these are also concatenated. Spikes
    occurring outside the spacing are dropped.

    Returns a single dict with {stim, duration, spikes}

    """
    n_before = int(pad_before / dt)
    n_after = int(pad_after / dt)
    padded_stims = []
    clipped_spikes = []
    duration = 0
    for i, d in enumerate(seq):
        s = d["stim"]
        nf, nt = s.shape
        fv_before = fill_value or s[:, 0].mean()
        p_before = fv_before * np.ones((nf, n_before), dtype=s.dtype)
        fv_after = fill_value or s[:, -1].mean()
        p_after = fv_after * np.ones((nf, n_after), dtype=s.dtype)
        padded_stims.extend((p_before, s, p_after))

        duration += pad_before
        newtl = tl.offset(tl.subrange(d["spikes"], -pad_before, d["duration"] + pad_after), -duration)
        clipped_spikes.append(tuple(newtl))
        duration += (d["duration"] + pad_after)

    return {"stim": np.concatenate(padded_stims, axis=1),
            "spikes": list(tl.merge(*clipped_spikes)),
            "duration": duration}
