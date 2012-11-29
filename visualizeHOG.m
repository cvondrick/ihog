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
hist(feat(:));
title('HOG distribution');
