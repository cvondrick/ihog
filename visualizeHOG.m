% visualizeHOG(feat)
%
% This function provides a diagnostic visualization of a HOG feature.  This is
% meant to be a drop-in replacement for the voc-release5 framework.

function visualizeHOG(feat, verbosity),

if ~exist('verbosity', 'var'),
  verbosity = 0;
end

if verbosity == 0,
  nfigs = 1;
elseif verbosity == 1,
  nfigs = 2;
else,
  nfigs = 3;
end

im = invertHOG(max(feat, 0));

if min(feat(:)) < 0,
  buff = 5;
  neg = invertHOG(max(-feat, 0));
  neg = padarray(neg, [buff buff], 0.5, 'both');
  im = padarray(im, [buff buff], 0.5, 'both');

  im = [im neg];
end

clf;

subplot(nfigs,2,1);
showHOG(feat);
title('HOG');

subplot(nfigs,2,2);
imagesc(im);
axis image;
colormap gray;
title('Inverse');

if nfigs == 1,
  return;
end

subplot(nfigs,2,3);
showHOG(feat - mean(feat(:)));
title('0-mean HOG');

subplot(nfigs,2,4);

if min(feat(:)) < 0,
  buff = 5;
  pos = HOGtexture(max(feat, 0));
  pos = padarray(pos, [buff buff], 0.5, 'both');
  neg = HOGtexture(max(-feat, 0));
  neg = padarray(neg, [buff buff], 0.5, 'both');
  imagesc([pos neg]);
else,
  imagesc(HOGtexture(feat));
end

axis image;
title('HOG Texture');

if nfigs == 2,
  return;
end

subplot(nfigs,2,5);
f = zeros(size(feat));
f(:, :, 19:27) = feat(:, :, 1:9);
showHOG(f);
title('Positive Signed HOG');

subplot(nfigs,2,6);
f = zeros(size(feat));
f(:, :, 19:27) = feat(:, :, 10:18);
showHOG(f);
title('Negative Signed HOG');



% HOGtexture(feat)
%
% Creates a visualization of the HOG texture features. For each HOG cell, it builds a histogram
% and attempts to visualize it spatially.
function bigfig = HOGtexture(feat),
texture = feat(:, :, 9*3+1:end);
[ny,nx,nf] = size(texture);

for k=1:nf,
  t = texture(:, :, k);
  texture(:, :, k) = texture(:, :, k) - min(t(:));
  t = texture(:, :, k);
  m = max(t(:));
  if m > 0,
    texture(:, :, k) = texture(:, :, k) / m;
  end
end

b = 1;
h = 8 * 3;
w = round(8 / nf) * 3;
bigfig = ones(ny*(h+b), nx*(w*nf+b), 3) * 0.75;
cc = prism(nf);

for i=1:ny,
  for j=1:nx,
    for k=1:nf,
      val = round(texture(i, j, k) * h);
      for z=1:3,
        bigfig((i-1)*(h+b)+1+(h+b-val) : i*(h+b), ...
               (j-1)*(w*nf+b)+1+(k-1)*w+1 : (j-1)*(w*nf+b)+1+k*w, z) = cc(k, z);
      end
    end
  end
end
