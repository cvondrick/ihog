% visualizeHOG(feat)
%
% This function provides a diagnostic visualization of a HOG feature.  This is
% meant to be a drop-in replacement for the voc-release5 framework.
%
% Usage is simple:
%   >> feat = features(im, 8);
%   >> visualizeHOG(feat);
%
% and the current figure will contain both the standard HOG glyph visualization as well
% as the inverse.  This function has extra verbosity outputs you can use too:
%
%   >> visualizeHOG(feat, 0);
%

function visualizeHOG(feat, verbosity),

if ~exist('verbosity', 'var'),
  verbosity = 0;
end

if verbosity == 0,
  nfigs = 1;
else,
  nfigs = 2;
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
