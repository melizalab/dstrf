# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Simple parametric filters"""
from __future__ import print_function, division, absolute_import

import numpy as np


def exponential(tau, amplitude, duration, dt):
    """ An exponential decay with time constant tau """
    tt = np.arange(0, duration, dt)
    return amplitude * 1 / tau * np.exp(-tt / tau), tt


def alpha(tau, amplitude, duration, dt):
    """An alpha function kernel with time constant tau"""
    tt = np.arange(0, duration, dt)
    return (amplitude * tt / tau * np.exp(-tt / tau), tt)


def gammadiff(tau1, tau2, amplitude, duration, dt):
    """ Difference of gamma function kernel """
    from scipy.special import gamma
    tt = np.arange(0, duration, dt)
    kg1 = 1 / (gamma(6) * tau1) * (tt / tau1)**5 * np.exp(-tt / tau1)
    kg2 = 1 / (gamma(6) * tau2) * (tt / tau2)**5 * np.exp(-tt / tau2)
    kg = kg1 - kg2 / 1.5
    return (kg / np.linalg.norm(kg) * amplitude, tt)


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
