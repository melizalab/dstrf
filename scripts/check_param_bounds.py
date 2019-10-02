# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" This script does a quick check to see if parameter estimates are in bounds """
from __future__ import print_function, division

import argparse
from dstrf import models
import numpy as np
from munch import Munch


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="check if spike history parameter estimates are in bounds")
    p.add_argument("config", help="path to configuration yaml file")
    p.add_argument("fitfile", help="path to output npz file(s)", nargs="+")

    args = p.parse_args()
    with open(args.config, "rt") as fp:
        cf = Munch.fromYAML(fp)

    matboundprior = models.matbounds(cf.model.ataus[0], cf.model.ataus[1], cf.model.t_refract)

    for path in args.fitfile:
        results = np.load(path)
        if "samples" in results:
            pos = results["samples"]
            allowed = [matboundprior(s) for s in pos]
            print("{}\t{}/{} samples in allowed region".format(path, sum(allowed), len(allowed)))
        else:
            w0 = results["mle"]
            if matboundprior(w0):
                print("{}\tOK".format(path))
            else:
                print("{}\tout of bounds".format(path))
