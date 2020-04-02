# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" This script will do assimilation from simulated data """
from __future__ import print_function, division

import os
import sys
import argparse
import json
import numpy as np
from munch import Munch
import emcee
from emcee_tools import priors, utils, startpos
from dstrf import io, data, simulate, models, strf, mle, util


def xvalidate(mlest, cf, **kwargs):
    """ Use cross-validation to find optimial l1 and l2 regularization params

    - mlest: initialized maximum likelihood estimator
    - cf: configuration Munch

    Returns (rf_alpha, rf_lambda), loglike, best_params
    """
    import time
    from dstrf import crossvalidate
    import progressbar as pb

    gtol = cf.xvalidate.get("gtol", 1e-5)
    early_stop = cf.xvalidate.get("early_stop", False)
    l1_ratios = cf.xvalidate.l1_ratios
    reg_grid = np.logspace(cf.xvalidate.grid.lower, cf.xvalidate.grid.upper, cf.xvalidate.grid.count)[::-1]
    scores = []
    results = []

    steps = len(l1_ratios) * len(reg_grid)
    print(" - expected number of steps: {}".format(steps))
    if early_stop:
        print(" - will stop if score decreases for more than {} steps".format(early_stop))
    hfmt = "{:>5}  {:>8}  {:>8}  {:>15}  {:>10}  {:>10}"
    dfmt = "{:>5}  {:>8.2g}  {:>8.2g}  {:>15.5g} {:>10}  {:>10}"
    hdr = hfmt.format("step", "α", "λ", "likelihood", "time", "ETA")
    print(hdr)
    print("-" * len(hdr))
    start = time.time()
    for step, reg, s, w in crossvalidate.elasticnet(mlest, 4, reg_grid, l1_ratios, gtol=gtol,
                                                    early_stop=early_stop, **kwargs):
        now = time.time()
        eta = ((now - start) / (step + 1)) * (steps - step - 1)
        scores.append(s)
        results.append((reg, s, w))
        print(dfmt.format(step, reg[0], reg[1], s, pb.Timer.format_time(now - start), pb.Timer.format_time(eta)),
              flush=True)

    best_idx = np.argmax(scores)
    return results[best_idx]


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="assimilate GLM model to simulated or real spike data")
    p.add_argument("--restart", "-r", help="start with parameters from output of previous run")
    p.add_argument("--constrained", action="store_true", help="use constrained maximum likelihood estimator")
    p.add_argument("--xval", "-x", action="store_true", help="use cross-validation to optimize regularization params")
    p.add_argument("--mcmc", "-m", action="store_true", help="use MCMC to sample from posterior distribution")
    p.add_argument("--save-chain", action="store_true", help="for MCMC, store complete chain in output file")
    p.add_argument("--save-data", action="store_true", help="store assimilation data in output file")
    p.add_argument("--update-config", "-k",
                   help="set configuration parameter. Use JSON literal. example: -k data.filter.rf=20",
                   action=util.ParseKeyVal, default=dict(), metavar="KEY=VALUE")
    p.add_argument("--skip-completed", "-s", action="store_true", help="skip run if output file exists")
    p.add_argument("config", help="path to configuration yaml file")
    p.add_argument("outfile", help="path to output npz file")

    args = p.parse_args()
    with open(args.config, "rt") as fp:
        cf = Munch.fromYAML(fp)

    if args.skip_completed and os.path.exists(args.outfile):
        print("skipping run - output file {} already exists".format(args.outfile))
        sys.exit(0)

    for k, v in args.update_config.items():
        path = k.split(".")
        util.assoc_in(cf, path, v)

    model_dt = cf.model.dt
    ncos = cf.model.filter.ncos
    kcosbas = strf.cosbasis(cf.model.filter.len, ncos)

    print("loading/generating data using", cf.data.source)
    stim_fun = getattr(data, cf.data.source)
    data     = stim_fun(cf)

    try:
        p_test = cf.data.test.proportion
    except AttributeError:
        p_test = None
    assim_data = io.subselect_data(data, p_test)

    if "model" in cf.data:
        print("simulating response using {}".format(cf.data.model))
        data_fun = getattr(simulate, cf.data.model)
        assim_data = data_fun(cf, assim_data)
    data = io.merge_data(assim_data)
    print(" - duration:", data["duration"])
    print(" - stim bins:", data["stim"].shape[1])
    print(" - spike bins:", data["spike_v"].shape[0])
    print(" - total spikes:", np.sum(data["spike_v"]))

    # this always fails on the first try for reasons I don't understand
    try:
        mlest = mle.mat(data["stim"], kcosbas, data["spike_v"], data["spike_h"], data["stim_dt"], data["spike_dt"])
    except TypeError:
        pass
    krank = cf.model.filter.get("rank", None)
    if krank is None:
        print("receptive field is full rank")
        mlest = mle.mat(data["stim"], kcosbas, data["spike_v"], data["spike_h"], data["stim_dt"], data["spike_dt"])
    else:
        print("receptive field is rank {}".format(krank))
        mlest = mle.matfact(data["stim"], kcosbas, krank, data["spike_v"], data["spike_h"], data["stim_dt"], data["spike_dt"])

    if args.constrained:
        print("setting up parameter inequality constraints")
        nparams = 1 + mlest.n_hparams + mlest.n_kparams
        constraint = models.matconstraint(nparams, cf.model.ataus[0], cf.model.ataus[1], cf.model.t_refract)
        optargs = {"method": "trust-constr", "constraints": [constraint]}
    else:
        print("using unconstrained optimization")
        optargs = {}

    if args.xval:
        print("cross-validating to find optimal regularization parameters")
        (rf_alpha, rf_lambda), loglike, w0 = xvalidate(mlest, cf, **optargs)
    elif args.restart:
        print("restarting from parameter estimates in {}".format(args.restart))
        results = np.load(args.restart)
        rf_lambda = results["reg_lambda"]
        rf_alpha = results["reg_alpha"]
        w0 = results["mle"]
    else:
        print("finding maximum likelihood estimate")
        try:
            rf_lambda = cf.model.prior.l1
        except AttributeError:
            rf_lambda = 0.0
        try:
            rf_alpha = cf.model.prior.l2
        except AttributeError:
            rf_alpha = 0.0
        w0 = mlest.estimate(reg_alpha=rf_alpha, reg_lambda=rf_lambda, **optargs)
    print(" - regularization params: alpha={:2}, lambda={:2}".format(rf_alpha, rf_lambda))
    print(" - MLE rate and adaptation parameters:", w0[:3])

    out = {"mle": w0, "reg_alpha": rf_alpha, "reg_lambda": rf_lambda}
    if args.mcmc:
        # this code has to be at top level so that lnpost can be pickled (an
        # emcee requirement)
        print("sampling from the posterior")
        print(" - walkers: {}".format(cf.emcee.nwalkers))
        if sys.platform == 'darwin':
            cf.emcee.nthreads = 1

        # set up priors - base rate and adaptation
        mat_prior = priors.joint_independent([priors.uniform(l, u) for (l, u) in cf.emcee.bounds])
        # additional constraint to stay out of disallowed region
        matboundprior = models.matbounds(cf.model.ataus[0], cf.model.ataus[1], cf.model.t_refract)

        def lnpost(theta):
            """Posterior probability"""
            mparams = theta[:3]
            if not matboundprior(mparams):
                return -np.inf
            lp = mat_prior(mparams)
            if not np.isfinite(lp):
                return -np.inf
            # mlest can do penalty for lambda
            ll = mlest.loglike(theta, rf_lambda, rf_alpha)
            if not np.isfinite(ll):
                return -np.inf
            return lp - ll

        sampler = emcee.EnsembleSampler(cf.emcee.nwalkers, w0.size, lnpost,
                                        threads=cf.emcee.nthreads)

        # initial state is a gaussian ball around the ML estimate
        step = 0
        if args.restart and "samples" in results:
            print(" - restarting from samples in {}".format(args.restart))
            pos = results["samples"]
            prob = results["prob"]
            try:
                step = results["step"]
            except KeyError:
                pass
            print(" - last step was {}".format(step))
        else:
            print(" - initializing walkers around point estimate")
            p0 = startpos.normal_independent(cf.emcee.nwalkers, w0, np.abs(w0) * cf.emcee.startpos_scale)
            # p0 = startpos.normal_independent(cf.emcee.nwalkers, w0, 1e-4)
            nburnin = cf.emcee.get("nburnin", 50)
            print(" - burn-in sampler for {} steps".format(nburnin))
            tracker = utils.convergence_tracker(nburnin, skip=25, start=step)
            for step, pos, prob, _ in tracker(sampler.sample(p0, storechain=False, iterations=nburnin)):
                continue
            sampler.reset()

        print(" - replacing zero-probability chains")
        utils.replace_invalid_walkers(pos, prob)

        tracker = utils.convergence_tracker(cf.emcee.nsteps, skip=25, start=step)
        print(" - begin sampling: {} steps".format(cf.emcee.nsteps))
        for step, pos, prob, _ in tracker(sampler.sample(pos,
                                                         lnprob0=prob,
                                                         storechain=args.save_chain,
                                                         iterations=cf.emcee.nsteps)):
            continue

        if args.save_chain:
            gr = utils.gelman_rubin(sampler.chain[:, -200:, :])
            #print(" - max autocorrelation: {:3}".format(sampler.acor.max()))
            print(" - average acceptance fraction: {:.2%}".format(sampler.acceptance_fraction.mean()))
            print(" - average Gelman-Rubin statistic (last 200 steps): {:.2}".format(gr.mean()))
            out["chain"] = sampler.chain

        print(" - lnpost of p median: {}".format(np.median(prob)))
        w0 = np.median(pos, 0)
        print("MAP rate and adaptation parameters:", w0[:3])
        out["samples"] = pos
        out["prob"] = prob
        out["step"] = step


    if args.save_data:
        print("saving assimilation data in output archive")
        out.update(stim=data["stim"],
                   spike_v=data["spike_v"], spike_h=data["spike_h"], duration=data["duration"])
        if "model" in cf.data:
            k1 = simulate.get_filter(cf)[0]
            out.update(kernel=k1)


    np.savez(args.outfile, **out)
