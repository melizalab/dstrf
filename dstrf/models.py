# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""dstrf models"""
from __future__ import print_function, division, absolute_import

import numpy as np


def matbounds(t1, t2, tr):
    """Returns function for boundaries on adaptation parameters based on disallowed
    region (see Yamauchi et al 2011)

    """
    aa1 = -(1 - np.exp(-tr / t2)) / (1 - np.exp(-tr / t1))
    aa2 = -(np.exp(tr / t2) - 1) / (np.exp(tr / t1) - 1)
    return lambda mparams: (mparams[2] > aa1 * mparams[1]) and (mparams[2] > aa2 * mparams[1])


def predict_spikes_glm(V, params, cf):
    """Predict spikes using GLM from voltage and rate/adaptation params"""
    import mat_neuron._model as mat
    upsample = int(cf.data.dt / cf.model.dt)
    omega, a1, a2  = params
    return mat.predict_poisson(V - omega, (a1, a2),
                               cf.model.ataus, cf.model.t_refract, cf.model.dt, upsample)
