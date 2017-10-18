# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for file IO"""
from __future__ import print_function, division, absolute_import

import os
import glob
import toelis as tl

def load_crcns(cell, stim_type, root="/home/data/crcns/", window, step, **specargs):
    """Load data from CRCNS repository

    Additional keyword arguments are passed to load_stimulus()
    """
    spikesroot = os.path.join(root, "all_cells", cell, stim_type)
    stimroot = os.path.join(root, "all_stims")

    out = []
    for fname in glob.iglob("*.toe_lis"):
        base, ext = os.path.splitext(fname)
        spec, dur = load_stimulus(os.path.join(stimroot, base + ".wav"), window, step, **specargs)
        spikes = tl.read(open(fname, "rt"))
        out.append({"cell_name": cell,
                    "stim_name": base,
                    "duration": dur,
                    "stim": spec,
                    "stim_dt": step,
                    "spikes": spikes})
    return out


def load_stimulus(path, windowtime, step, f_min=0.5, f_max=8.0, f_count=30,
                  gammatone=False):
    """Load sound stimulus and calculate spectrotemporal representation.

    Parameters:

    path: location of wave file
    windowtime: duration of window (in ms)
    step: window step (in ms)
    f_min: minimum frequency (in kHz)
    f_max: maximum frequency (in kHz)
    f_count: number of frequency bins
    gammatone: if True, use gammatone filterbank

    Returns spectrogram, duration (ms)
    """
    import ewave
    with ewave.open(path) as fp:
        Fs = fp.sampling_rate / 1000
        duration = fp.nframes / Fs
        osc = fp.read()
        if gammatone:
            import gammatone.gtgram as gg
            Pxx = gg.gtgram(osc, Fs, windowtime, step, f_count, f_min, f_max)
        else:
            import libtfr
            # nfft based on desired number of channels btw f_min and f_max
            nfft = f_count / (f_max - f_min) * Fs
            npoints = int(Fs * windowtime)
            if nfft < npoints:
                raise ValueError("window size {} ({} points) too small "
                                 "for desired freq resolution".format(windowtime, npoints))

            nstep = int(Fs * step)
            taper = np.hanning(npoints)
            mfft = libtfr.mfft_precalc(nfft, taper)
            Pxx = mfft.stft(wave, nstep)
            freqs, ind = libtfr.fgrid(Fs, nfft, [f_min, f_max])
            Pxx = Pxx[ind,:]
        return Pxx, duration
