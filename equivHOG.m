function out = equivHOG(orig, n, gam, sig, pd),

if ~exist('pd', 'var'),
  pd = load('pd.mat');
end

orig = im2double(orig);
feat = computeHOG(orig, 8);

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
numwindows = (ny+12-pd.ny+1)*(nx+12-pd.nx+1);

fprintf('ihog: attempting to find %i equivalent images in HOG space\n', n);

prev.a = zeros(0, 0, 0);
prev.gam = gam;
prev.sig = sig;

ims = ones((ny+2)*8, (nx+2)*8, n);
hogs = zeros(ny, nx, nf, n);
hogdists = zeros(n, 1);

for i=1:n,
  fprintf('ihog: searching for image %i of %i\n', i, n);
  [im, prev] = invertHOG(feat, pd, prev);

  ims(:, :, i) = mean(im, 3);
  hogs(:, :, :, i) = computeHOG(im, 8);

  d = hogs(:, :, :, i) - feat;
  hogdists(i) = sqrt(mean(d(:).^2));

  subplot(122);
  imdiffmatrix(ims(:, :, 1:i), orig, 5);

  subplot(321);
  sparsity = mean(reshape(double(prev.a(:, :, 1:i) == 0), [], i));
  plot(sparsity(:), '.-', 'LineWidth', 2, 'MarkerSize', 40);
  title('Alpha Sparsity');
  ylabel('Sparsity');
  ylim([0.75 1]);
  grid on;

  subplot(323);
  plot(hogdists(1:i), '.-', 'LineWidth', 2, 'MarkerSize', 40);
  title('HOG Distance to Target');
  ylim([0 0.4]);
  grid on;

  subplot(325);
  imagesc(hogimvis(ims(:, :, 1:i), hogs(:, :, :, 1:i)));
  axis image;

  colormap gray;
  drawnow;
end

out = ims;



function out = hogimvis(ims, hogs),

out = [];
for i=1:size(ims,3),
  im = ims(:, :, i);
  hog = hogs(:, :, :, i);
  hog(:) = max(hog(:) - mean(hog(:)), 0);
  hog = showHOG(hog);
  hog = imresize(hog, size(im));
  hog(hog > 1) = 1;
  hog(hog < 0) = 0;
  im = padarray(im, [5 10], 1);
  hog = padarray(hog, [5 10], 1);
  graphic = [im; hog];
  out = [out graphic];
end
out = padarray(out, [5 0], 1);
