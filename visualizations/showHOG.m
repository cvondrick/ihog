% showHOG(w)
%
% Legacy HOG visualization
function out = showHOG(w)

% Make pictures of positive and negative weights
bs = 20;
pos = HOGpicture(w, bs);
neg = HOGpicture(-w, bs);

% Put pictures together and draw
buff = 10;
if min(w(:)) < 0
  pos = padarray(pos, [buff buff], 0.5, 'both');
  neg = padarray(neg, [buff buff], 0.5, 'both');
else
  im = pos;
end

im(im < 0) = 0;
im(im > 1) = 1;

if nargout == 0,
  imagesc(im); 
  colormap gray;
  axis image;
else,
  out = im;
end
