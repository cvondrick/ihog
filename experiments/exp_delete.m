function exp_delete(),

param.mode = 'delete';

outpath = '/data/vision/torralba/hallucination/icnn/experiments/delete';

method = @(feat, pd, n, param, w, orig) equivCNN(feat, pd, n, param, w, orig);

exp_driver(param, outpath, method);
