# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Simple parametric filters"""
from __future__ import print_function, division, absolute_import

import numpy as np


def exponential(tau, amplitude, duration, dt):
    """ An exponential decay with time constant tau """
    tt = np.arange(0, duration, dt)
    ke = amplitude * 1 / tau * np.exp(-tt / tau)
    return ke[::-1], -tt[::-1]


def alpha(tau, amplitude, duration, dt):
    """An alpha function kernel with time constant tau"""
    tt = np.arange(0, duration, dt)
    ka = amplitude * tt / tau * np.exp(-tt / tau)
    return (ka[::-1], -tt[::-1])


def gammadiff(tau1, tau2, amplitude, ntau, dt, **kwargs):
    """ Difference of gamma function kernel """
    from scipy.special import gamma
    tt = np.arange(0, ntau * dt, dt)
    xtau1 = ntau * dt / tau1
    xtau2 = ntau * dt / tau2
    kg1 = 1 / (gamma(6) * xtau1) * (tt / xtau1)**5 * np.exp(-tt / xtau1)
    kg2 = 1 / (gamma(6) * xtau2) * (tt / xtau2)**5 * np.exp(-tt / xtau2)
    kg = kg1 - kg2 / 1.5
    kg *= amplitude / np.linalg.norm(kg)
    return (kg[np.newaxis, ::-1], -tt[::-1])


def strf(nfreq, ntau, f_max, f_peak, t_peak, ampl, f_sigma, t_sigma, f_alpha, t_alpha, **kwargs):
    """Construct a parametric (mexican hat) STRF

    nfreq: resolution of the filter in pixels
    ntau: time window of the filter in ms
    f_max: maximum frequency of the signal
    f_peak: center frequency for the filter
    t_peak: offset between stimulus and response in ms (range: 0 to ntau)
    ampl:   amplitude of the wavelet peak
    f_sigma: width of the filter in the frequency axis -- bigger gamma = narrower frequency band
    t_sigma: width of the filter in the time axis -- bigger sigma = narrower time band
    f_alpha: depth of inhibitory sidebands on frequency axis -- bigger f_alpha = deeper sidebands
    t_alpha: depth of inhibitory sidebands on time axis -- bigger t_alpha = deeper sidebands
    Returns the RF (nfreq, ntau), tau (ntau,), and f (nfreq,)
    """
    ntau -= 1
    scale = nfreq / 50.0
    t = np.arange(float(np.negative(ntau)), 1)
    tscale = np.arange(np.negative(ntau), 1, 2)
    x = t_peak
    f = np.arange(0, f_max + 1, float(f_max) / nfreq)
    y = f_peak
    tc = t + x
    fc = f - y
    tprime, fprime = np.meshgrid(tc, fc)
    t_sigma = t_sigma / scale
    Gtf = (ampl * np.exp(-t_sigma**2 * tprime**2 - f_sigma**2 * fprime**2) *
           (1 - t_alpha**2 * t_sigma**2 * tprime**2) * (1 - f_alpha**2 * f_sigma**2 * fprime**2))
    return Gtf, tscale, f


def hg(nfreq, ntau, f_max, f_peak, t_max, t_peak, ampl, t_sigma, f_sigma, t_omega, f_omega, Pt, Pf, **kwargs):
    """Construct an unrotated gabor STRF

    nfreq: number of frequency channels in the filter
    ntau: number of time points in the filter
    f_max: maximum frequency of the signal (in Hz)
    f_peak: center frequency for the filter (in Hz)
    t_max: duration of the filter (in ms)
    t_peak: offset between stimulus and response in ms (range: 0 to -t_max)
    ampl:   amplitude of the wavelet peak
    t_sigma: width of the filter in the time axis (in ms)
    f_sigma: width of the filter in the frequency axis (in Hz)
    t_omega: temporal modulation frequency
    f_omega: spectral modulation frequency
    Pt: temporal phase
    Pf: spectral phase
    Returns the RF (nfreq, ntau), tau (ntau,), and f (nfreq,)
    """
    dt = ntau / t_max
    df = nfreq / f_max
    t_sigma *= dt
    t_omega /= dt
    f_sigma *= df
    f_omega /= df
    t = np.arange(0, ntau) * dt
    x = t_peak * dt
    f = np.arange(0, nfreq)
    y = f_peak * df
    tc = t + x
    fc = f - y
    tprime, fprime = np.meshgrid(tc, fc)
    H = np.exp(-0.5 * ((tprime) / t_sigma)**2) * np.cos(2 * np.pi * t_omega * (tprime) + Pt)
    G = np.exp(-0.5 * ((fprime) / f_sigma)**2) * np.cos(2 * np.pi * f_omega * (fprime) + Pf)
    strf = H * G
    strf /= np.sqrt(np.trapz(np.trapz(strf**2, axis=1)))
    return np.fliplr(ampl * strf), -t[::-1], f


def gabor(nfreq, ntau, f_max, f_peak, t_peak, ampl, f_sigma, t_sigma, theta, lmbda, psi, **kwargs):
    """Construct a parametric (gabor) STRF

    nfreq: resolution of the filter in pixels
    ntau: time window of the filter in ms
    f_max: maximum frequency of the signal
    f_peak: center frequency for the filter
    t_peak: offset between stimulus and response in ms (range: 0 to ntau)
    ampl:   amplitude of the wavelet peak
    f_sigma: width of the filter in the frequency axis -- bigger = narrower frequency band
    t_sigma: width of the filter in the time axis -- bigger = narrower time band
    ...
    Returns the RF (nfreq, ntau), tau (ntau,), and f (nfreq,)
    """
    ntau -= 1
    t = np.arange(float(np.negative(ntau)), 1)
    tscale = np.arange(np.negative(ntau), 1, 2)
    x = t_peak
    f = np.arange(0, f_max + 1, float(f_max) / nfreq)[:nfreq]
    y = f_peak
    tc = t + x
    fc = f - y
    tprime, fprime = np.meshgrid(tc, fc)
    # Rotation
    x_theta = tprime * np.cos(theta) + fprime * np.sin(theta)
    y_theta = -tprime * np.sin(theta) + fprime * np.cos(theta)
    gb = ampl * np.exp(-.5 * (x_theta ** 2 / t_sigma ** 2 + y_theta ** 2 / f_sigma ** 2)) * np.cos(2 * np.pi / lmbda * x_theta + psi)
    return (gb, tscale, f)
