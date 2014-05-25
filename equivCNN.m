function out = equivCNN(feat, pd, n, param, w),

if ~exist('pd', 'var'),
  pd = load('pd-cnn.mat');
end

if ~exist('n', 'var'),
  n = 6;
end
if ~exist('param', 'var'),
  param = struct();
end
if ~exist('w', 'var'),
  w = ones(pd.featdim, 1);
end

bord = 5;
[ny, nx, nf] = size(feat);

fprintf('icnn: attempting to find %i equivalent images in CNN space\n', n);

ims = ones([pd.imdim n]);

prev = param;

for i=1:n,
  fprintf('icnn: searching for image %i of %i\n', i, n);
  [im, prev] = invertCNN(feat, pd, prev, w);

  ims(:, :, :, i) = im;

  imdiffmatrix(ims(:, :, :, 1:i));
  drawnow;
end

out = ims;
