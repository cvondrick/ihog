% interesting at:
%   slice=1, gam=.000001

function exp_rgb(gam, slices),

param.mode = 'rgb';
param.gam = gam;
param.slices = slices;

outpath = sprintf('/data/vision/torralba/hallucination/icnn/experiments/%s_gam=%0.8f_slices=%i', param.mode, param.gam, param.slices);

method = @(feat, pd, n, param, w, orig) equivCNN(feat, pd, n, param, w, orig);

exp_driver(param, outpath, method);
