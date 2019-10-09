# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" This script does a quick check of spike data; useful for excluding nonresponsive neurons """
from __future__ import print_function, division

import argparse
import numpy as np
from munch import Munch

from dstrf import data, performance, util, io

if __name__ == "__main__":

    import argparse

    p = argparse.ArgumentParser(description="calculate statistics of spiking data")
    p.add_argument("--binsize", "-b", type=float, default=10, help="bin size for PSTH (in ms)")
    p.add_argument("config", help="path to configuration yaml file")
    p.add_argument("cells", help="name(s) of the cell", nargs="+")

    args = p.parse_args()
    with open(args.config, "rt") as fp:
        cf = Munch.fromYAML(fp)
    stim_fun = getattr(data, cf.data.source)

    print("cell\tduration\tspikes\trate\teo.cc")
    for cell in args.cells:
        try:
            cf.data.cell = cell
            raw_data = stim_fun(cf)
            data = io.merge_data(raw_data)
            upsample = int(args.binsize / cf.model.dt)
            eo = performance.corrcoef(data["spike_v"][::2], data["spike_v"][1::2], upsample, 1)
            rate = 1000 * np.sum(data["spike_v"]) / data["duration"] / data["spike_v"].shape[1]
            print("{}\t{}\t{}\t{:3.3}\t{:3.3}".format(cell,
                                                      data["duration"],
                                                      np.sum(data["spike_v"]),
                                                      rate,
                                                      eo))
        except Exception as e:
            print("{}\terror:{}".format(cell, e))
