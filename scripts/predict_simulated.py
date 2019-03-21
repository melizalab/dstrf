# -*- coding: utf-8 -*-
# -*- mode: python -*-

             astim=data["stim"], aspikes=data["spike_v"],
             pos=pos, prob=prob, eo=eo, cc=cc,
             tstim=tdata["stim"], tspikes=tdata["spike_v"],
             pspikes=np.column_stack(pred_spikes))


    print("simulating response for testing using {}".format(cf.data.model))
    tdata = io.merge_data(data_fun(cf, test_data, random_seed=1000))

    # we use the estimator to generate predictions
    if krank is None:
        mltest = mle.mat(tdata["stim"], kcosbas, tdata["spike_v"], tdata["spike_h"], tdata["stim_dt"], tdata["spike_dt"])
    else:
        mltest = mle.matfact(tdata["stim"], kcosbas, krank, tdata["spike_v"], tdata["spike_h"], tdata["stim_dt"], tdata["spike_dt"])

    tspike_v = tdata["spike_v"]
    pred_spikes = np.zeros_like(tspike_v)
    samples = np.random.permutation(cf.emcee.nwalkers)[:cf.data.trials]
    for i, idx in enumerate(samples):
        sample = pos[idx]
        V = mltest.V(sample)
        S = models.predict_spikes_glm(V, sample[:3], cf)
        pred_spikes[:, i] = S

    upsample = int(cf.data.dt / cf.model.dt)
    test_psth = spikes.psth(tspike_v, upsample, 1)
    pred_psth = spikes.psth(pred_spikes, upsample, 1)

    eo = performance.corrcoef(tspike_v[::2], tspike_v[1::2], upsample, 1)
    cc = np.corrcoef(test_psth, pred_psth)[0, 1]
    print("EO cc: %3.3f" % eo)
    print("pred cc: %3.3f" % cc)
    print("spike count: data = {}, pred = {}".format(tspike_v.sum(), pred_spikes.sum()))
