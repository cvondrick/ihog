% visualizeColorize(im, pd)
%
% Attempts to colorize the image using HOG inversion.
function out = visualizeColorize(im, pd)

feat = features(im, 8);
ihog = invertHOG(feat, pd);

im = imresize(im, [size(ihog, 1) size(ihog, 2)]);
im(im > 1) = 1;
im(im < 0) = 0;

imhsv = rgb2hsv(im);

ihoghsv = rgb2hsv(ihog);
ihoghsv(:, :, 3) = imhsv(:, :, 3);

if nargout > 0,
  out = hsv2rgb(ihoghsv);
else,
  imagesc(hsv2rgb(ihoghsv)); 
  axis image;
end
