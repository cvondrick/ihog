% builds the baseline result files
function exp_baseline(mag),

param.mag = mag;

outpath = sprintf('/data/vision/torralba/hallucination/icnn/experiments/baseline_mag=%f', mag);

method = @(feat, pd, n, param, w, orig) nearbyCNN(feat, pd, n, param.mag, orig);

exp_driver(param, outpath, method);
