function exp_standard(gam),

param.mode = 'standard';
param.gam = gam;

outpath = sprintf('/data/vision/torralba/hallucination/icnn/experiments/standard_gam=%0.8f', param.gam);

method = @(feat, pd, n, param, w, orig) equivCNN(feat, pd, n, param, w, orig);

exp_driver(param, outpath, method);
