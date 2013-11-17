function im = invertCNN(feat, pd),

feat(:) = feat(:) - mean(feat(:));
feat(:) = feat(:) / (sqrt(sum(feat(:).^2) + eps));

% solve lasso problem
param.lambda = pd.lambda;
param.mode = 2;
param.pos = true;
a = full(mexLasso(single(feat'), pd.dhog, param));
recon = pd.dgray * a;

im = reshape(recon, [pd.ny pd.nx 3]);
im(:) = im(:) - min(im(:));
im(:) = im(:) / max(im(:));
