# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""dstrf models"""
from __future__ import print_function, division, absolute_import
import numpy as np

from dstrf import strf

class glmat(object):

    def __init__(self, nfreq, nrank, taus):
        self.nfreq = nfreq
        self.nrank = nrank
        self.taus = taus

    def predict(self, stim, rfparams, matparams):
        pass
