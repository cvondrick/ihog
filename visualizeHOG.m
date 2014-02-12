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
% as the inverse. 
%
% If 'feat' has negative values, a second row will appear of the negatives.

function out = visualizeHOG(feat, neg),

if ~exist('neg', 'var'),
  neg = true;
end

if neg,
  posfeat = max(feat, 0);
  negfeat = max(-feat, 0);
else,
  posfeat = feat;
  negfeat = zeros(size(feat));
end

s = [size(feat,1)*8+16 size(feat,2)*8+16];

im = invertHOG(posfeat);
hog = showHOG(max(0, posfeat));
hog = imresize(hog, s);
hog(hog > 1) = 1;
hog(hog < 0) = 0;

buff = 5;
im = padarray(im, [buff buff], 0.5, 'both');
hog = padarray(hog, [buff buff], 0.5, 'both');

if any(negfeat(:) > 0),
  hogneg = showHOG(max(0, negfeat));
  hogneg = imresize(hogneg, s);
  hogneg(hogneg > 1) = 1;
  hogneg(hogneg < 0) = 0;
  hogneg = padarray(hogneg, [buff buff], 0.5, 'both');

  neg = invertHOG(max(-feat, 0));
  neg = padarray(neg, [buff buff], 0.5, 'both');

  im = [im; neg];
  hog = [hog; hogneg];
end

im = [im hog];

if nargout == 0,
  imagesc(im);
  axis image;
  axis off;
  colormap gray;
else,
  out = im;
end
