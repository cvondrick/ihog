function out = equivCNN(feat, n, gam, sig, pd),

if ~exist('pd', 'var'),
  pd = load('pd-cnn.mat');
end

if ~exist('n', 'var'),
  n = 6;
end
if ~exist('gam', 'var'),
  gam = 100;
end
if ~exist('sig', 'var'),
  sig = 1;
end

bord = 5;
[ny, nx, nf] = size(feat);

fprintf('icnn: attempting to find %i equivalent images in CNN space\n', n);

prev.a = zeros(0, 0, 0);
prev.gam = gam;
prev.sig = sig;

ims = ones((ny+2)*8, (nx+2)*8, 3, n);

for i=1:n,
  fprintf('icnn: searching for image %i of %i\n', i, n);
  [im, prev] = invertCNN(feat, pd, prev);

  ims(:, :, :, i) = im;

  %subplot(211);
  imdiffmatrix(ims(:, :, :, 1:i));

  %subplot(212);
  %sparsity = mean(reshape(double(prev.a(:, :, 1:i) == 0), [], i));
  %plot(sparsity(:), '.-', 'LineWidth', 2, 'MarkerSize', 40);
  %title('Alpha Sparsity');
  %ylabel('Sparsity');
  %ylim([0.75 1]);
  %grid on;

  colormap gray;
  drawnow;
end

out = ims;
