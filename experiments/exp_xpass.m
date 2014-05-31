function exp_xpass(gam, sig),

param.mode = 'xpass';
param.gam = gam;
param.sig = sig;

outpath = sprintf('/data/vision/torralba/hallucination/icnn/experiments/xpass_gam=%0.8f_sig=%0.3f', param.gam, param.sig);

method = @(feat, pd, n, param, w, orig) equivCNN(feat, pd, n, param, w, orig);

exp_driver(param, outpath, method);
