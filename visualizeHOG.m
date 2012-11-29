% visualizeHOG(feat)
%
% This function provides a diagnostic visualization of a HOG feature.  This is
% meant to be a drop-in replacement for the voc-release5 framework.

function visualizeHOG(feat),

im = invertHOG(max(feat, 0));

if min(feat(:)) < 0,
  buff = 5;
  neg = invertHOG(max(-feat, 0));
  neg = padarray(neg, [buff buff], 0.5, 'both');
  im = padarray(im, [buff buff], 0.5, 'both');

  im = [im neg];
end

clf;

subplot(221);
showHOG(feat);
title('HOG');

subplot(222);
imagesc(im);
axis image;
colormap gray;
title('Inverse');

subplot(223);
showHOG(feat - mean(feat(:)));
title('0-mean HOG');

subplot(224);
imagesc(HOGtexture(feat));
axis image;
title('HOG Texture');


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

h = 8;
w = round(8 / nf);
bigfig = ones(ny*(h+1), nx*w*nf, 3) * 0.75;
cc = hsv(nf);

for i=1:ny,
  for j=1:nx,
    for k=1:nf,
      val = texture(i, j, k);
      val = round(val * h);
      color = cc(k, :);
      for z=1:3,
        bigfig((i-1)*(h+1)+1+(h+1-val):i*(h+1), ((j-1)*nf+k-1)*w+1:((j-1)*nf+k)*w, z) = color(z);
      end
    end
  end
end
