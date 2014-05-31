function exp_standard(gam),

pd = load('pd-caffe-hog.mat');
param.mode = 'hog';
param.gam = gam;

outpath = sprintf('/data/vision/torralba/hallucination/icnn/experiments/hog_gam=%0.8f', param.gam);

method = @(feat, pd, n, param, w, orig) equivCNN(feat, pd, n, param, w, orig);

exp_driver(param, outpath, method, pd);
