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

function visualizeHOG(feat),

s = [size(feat,1)*8+16 size(feat,2)*8+16];

im = invertHOG(max(feat, 0));
hog = HOGpicture(feat);
hog = imresize(hog, s);
hog(hog > 1) = 1;
hog(hog < 0) = 0;

buff = 5;
im = padarray(im, [buff buff], 0.5, 'both');
hog = padarray(hog, [buff buff], 0.5, 'both');

if min(feat(:)) < 0,
  hogneg = HOGpicture(-feat);
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

imagesc(im);
axis image;
axis off;
colormap gray;
