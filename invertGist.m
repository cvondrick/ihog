function im = invertGist(feat, pd),

% solve lasso problem
param.lambda = pd.lambda;
param.mode = 2;
a = full(mexLasso(single(feat'), pd.dhog, param));
recon = pd.dgray * a;

im = reshape(recon, [128 128]);
im(:) = im(:) - min(im(:));
im(:) = im(:) / max(im(:));
