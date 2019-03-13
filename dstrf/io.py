# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for file IO"""
from __future__ import print_function, division, absolute_import

import os
import glob
import json
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


def load_rothman(cell, root, window, step, load_internals=False, **specargs):
    """Load stimulus and response data from SNR4synhg

    Additional keyword arguments are passed to load_stimulus()
    """
    stimroot = os.path.join(root, "stims")

    out = []
    for fname in sorted(glob.iglob(os.path.join(root, cell, "spike*"))):
        stim_idx = int(fname[-2:])
        stim = "stim{}-0.wav".format(stim_idx)
        spec, dur = load_stimulus(os.path.join(stimroot, stim), window, step, **specargs)
        spikes = []
        for f in open(fname):
            spks = f.strip().split()
            spikes.append(np.asarray(spks, dtype='d'))
        celldata = {"cell_name": cell,
             "stim_name": stim,
             "duration": dur,
             "stim": spec,
             "stim_dt": step,
             "spikes": spikes}
        if load_internals:
            npzfile = "stim{}.npz".format(stim_idx)
            npzdata = np.load(os.path.join(root, cell, npzfile))
            celldata["I"] = npzdata['conv']
        out.append(celldata)
    return out


def load_rothman_rf(cell, root):
    """Load receptive field from rothman simulation dataset"""
    spikesroot = os.path.join(root, cell)
    rffile = np.load(os.path.join(spikesroot, "rf.npz"))
    return rffile['rf']



def load_neurobank(cell, window, step, **specargs):
    """ Load stimulus file and response data from neurobank repository """
    import itertools
    from nbank import find
    unitfile = next(find(cell, local_only=True))
    # first load and collate the responses, then load the stimuli
    out = []
    with open(unitfile, 'rU') as fp:
        data = json.load(fp)
        trials = sorted(data['pprox'], key = lambda x: (x['stimulus'], x['trial']))
        for stimname, trials in itertools.groupby(trials, lambda x: x['stimulus']):
            try:
                stimfile = next(find(stimname, local_only=True))
            except StopIteration:
                # no match with stimulus
                continue
            spec, dur = load_stimulus(stimfile, window, step, **specargs)
            out.append({"cell_name": cell,
                        "stim_name": stimname,
                        "duration": dur,
                        "stim": spec,
                        "stim_dt": step,
                        "spikes": [np.asarray(p["events"]) * 1000. for p in trials]})
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
    osc = ewave.rescale(fp.read(), 'h')
    if gammatone:
        import gammatone.gtgram as gg
        Pxx = gg.gtgram(osc, Fs * 1000, window / 1000, step / 1000, f_count, f_min * 1000, f_max * 1000)
    else:
        import libtfr
        # nfft based on desired number of channels btw f_min and f_max
        nfft = int(f_count / (f_max - f_min) * Fs)
        npoints = int(Fs * window)
        if nfft < npoints:
            raise ValueError("window size {} ({} points) too large for desired freq resolution {}. "
                             "Decrease to {} ms or increase f_count.".format(window, f_count,
                                                                             npoints, nfft / Fs))

        nstep = int(Fs * step)
        taper = np.hanning(npoints)
        mfft = libtfr.mfft_precalc(nfft, taper)
        Pxx = mfft.mtspec(osc, nstep)
        freqs, ind = libtfr.fgrid(Fs, nfft, [f_min, f_max])
        Pxx = Pxx[ind, :]
    if compress is not None:
        Pxx = np.log10(Pxx + compress) - np.log10(compress)
    return Pxx, Pxx.shape[1] * step


def pad_stimuli(data, before, after, fill_value=None):
    """Pad stimuli and adjust spike times in data

    Stimuli are usually preceded and followed by silent periods. This function
    pads the spectrograms with either by the specified fill_value, or by the
    average value in the first and last frame (if fill_value is None)

    Spike times are adjusted so that they are reference to the start of the
    padded stimulus, and all spike times outside the padded interval are
    dropped.

    - data: a sequence of dictionaries, which must contain 'spikes', 'stim' and
      'stim_dt' fields. This is modified in place
    - before: interval to pad before stimulus begins (in units of stim_dt)
    - after: interval to pad after stimulus ends

    NB: this needs to be run BEFORE preprocess_spikes as it will not touch
    spike_v or spike_h.

    """
    import toelis as tl
    for d in data:
        dt = d["stim_dt"]
        n_before = int(before / dt)
        n_after = int(after / dt)

        s = d["stim"]
        nf, nt = s.shape
        fv_before = s[:, 0].mean() if fill_value is None else fill_value
        p_before = fv_before * np.ones((nf, n_before), dtype=s.dtype)
        fv_after = s[:, -1].mean() if fill_value is None else fill_value
        p_after = fv_after * np.ones((nf, n_after), dtype=s.dtype)

        d["stim"] = np.c_[p_before, s, p_after]

        newtl = tl.offset(tl.subrange(d["spikes"], -before, d["duration"] + after), -before)
        d["spikes"] = list(newtl)
        d["duration"] += before + after
    return data


def preprocess_spikes(data, dt, sphist_taus):

    """Preprocess spike times in data

    Spike times are binned into intervals of duration dt. The times are
    then convolved with exponential kernels with amplitude 1.0 and time
    constants specified in sphist_taus. It's necessary to do this before merging
    stimuli to avoid having spike history carry over between stimuli (i.e., we
    should not assume that trial 1 on stimulus 1 immediately preceded trial 1 on
    stimulus 2).

    - data: a sequence of dictionaries, which must contain 'spikes', 'stim' and
      'stim_dt' fields.
    - dt: the duration of the step size for the model (same units as spike times)
    - sphist_taus: a sequence of time constants (same units as spike times)

    The following fields are added in place to the dictionaries in data:

    - spike_v: a 2-D binary array (bins, trials) giving the number of
      spikes in each bin
    - spike_h: a 3-D double array (bins, taus, trials) with the convolution
      of spike_v and exp(-t/tau)
    - spike_dt: the sampling interval

    """
    from mat_neuron._model import adaptation
    ntaus = len(sphist_taus)
    for d in data:
        ntrials = len(d["spikes"])
        nchan, nframes = d["stim"].shape
        nbins = nframes * int(d["stim_dt"] / dt)
        spike_v = np.zeros((nbins, ntrials), dtype='i')
        spike_h = np.zeros((nbins, ntaus, ntrials), dtype='d')
        for i, trial in enumerate(d["spikes"]):
            idx = (trial / dt).astype('i')
            spike_v[idx, i] = 1
            spike_h[:, :, i] = adaptation(spike_v[:, i], sphist_taus, dt)
        d["spike_v"] = spike_v
        d["spike_h"] = spike_h
        d["spike_dt"] = dt
    return data


def merge_data(seq):
    """Merge a sequence of stimuli into a single trial

    Takes a list or tuple of dicts containing {stim, stim_dt, spike_v, spike_h,
    spike_dt} and concatenates each of {stim, spike_spike_v, spike_h} along the
    appropriate axis, returning a single dictionary with {stim, stim_dt,
    spike_v, spike_h, spike_dt}

    """
    stim_dts = [d["stim_dt"] for d in seq]
    stim_dt = stim_dts[0]
    if not np.all(np.equal(stim_dts, stim_dt)):
        raise ValueError("not all stimuli have the same sampling rate")

    spike_dts = [d["spike_dt"] for d in seq]
    spike_dt = spike_dts[0]
    if not np.all(np.equal(spike_dts, spike_dt)):
        raise ValueError("not all spike vectors have the same sampling rate")

    ntrialss = [d["spike_v"].shape[1] for d in seq]
    ntrials = ntrialss[0]
    if not np.all(np.equal(ntrialss, ntrials)):
        raise ValueError("not all stimuli have the same number of trials")

    spike_v = np.concatenate([d["spike_v"] for d in seq], axis=0)
    return {
        "stim_dt": stim_dt,
        "spike_dt": spike_dt,
        "ntrials": ntrials,
        "stim": np.concatenate([d["stim"] for d in seq], axis=1),
        "spike_v": spike_v,
        "spike_t": [spk.nonzero()[0] for spk in spike_v.T],
        "spike_h": np.concatenate([d["spike_h"] for d in seq], axis=0),
        "duration": sum(d["duration"] for d in seq),
    }
