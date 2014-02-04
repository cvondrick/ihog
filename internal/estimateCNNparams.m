function [mu, sig, w] = estimateCNNparams(sourcedir, n, lambda),

if ~exist('lambda', 'var'),
  lambda = 500;
end
if ~exist('layer', 'var'),
  layer = 'pool5_cudanet_out';
end
if ~exist('featdim', 'var'),
  featdim = [6 6 256];
end

sig = zeros(prod(featdim));
mu = zeros(prod(featdim),1);

files = dir(sourcedir);
files = files(randperm(length(files)));
c = 0;
for i=1:length(files),
  if files(i).isdir,
    continue;
  end

  fprintf('icnn: load: %s (loaded %i of %i)\n', files(i).name, c, n);

  payload = load([sourcedir '/' files(i).name]);

  features = getfield(payload, layer);
  features = permute(features, [2 3 4 1]);
  features = reshape(features, [], size(features, 4));

  sig = sig + features * features';
  mu = mu + sum(features, 2);

  c = c + size(features, 2);

  if c > n,
    break;
  end
end

fprintf('icnn: computing mu and sigma\n');
mu = mu / c;
sig = sig / (c-1) - mu * mu';
sig = sig + lambda * eye(size(sig));

if nargout >= 3,
  fprintf('icnn: calculate whitening transformation\n');
  [v,d] = eig(sig);
  if any(diag(d) < eps),
    fprintf('icnn: warning! negative eigenvalues in whitening!\n');
  end
  w = v * diag(1./sqrt(diag(d))) * v';
end
