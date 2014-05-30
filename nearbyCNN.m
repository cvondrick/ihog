function out = nearbyCNN(feat, pd, n, mag, orig),

if ~exist('orig', 'var'),
  orig = [];
end

feat = repmat(feat, [1 n]);

noise = randn(size(feat));
noise = noise ./ repmat(sqrt(sum(noise.^2)), [pd.featdim 1]);
noise = noise * mag;
noise(:, 1) = 0;

feat = feat + noise;

out = invertCNN(feat, pd);
out = reclip(out, 0.02);

imdiffmatrix(out, orig);
