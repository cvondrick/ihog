% showHOG(w)
%
% Legacy HOG visualization
function out = showHOG(w)

% Make pictures of positive and negative weights
bs = 20;
pos = HOGpicture(w, bs);

% Put pictures together and draw
if min(w(:)) < 0
  buff = 10;
  neg = HOGpicture(-w, bs);

  pos = padarray(pos, [buff buff], 0.5, 'both');
  neg = padarray(neg, [buff buff], 0.5, 'both');

  im = [pos neg];
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



% HOGpicture(w)
%
% Make picture of positive HOG weights.
%   im = HOGpicture(w, bs)
function im = HOGpicture(w, bs)

if ~exist('bs', 'var'),
  bs = 20;
end

% construct a "glyph" for each orientaion
bim1 = zeros(bs, bs);
bim1(:,round(bs/2):round(bs/2)+1) = 1;
bim = zeros([size(bim1) 9]);
bim(:,:,1) = bim1;
for i = 2:9,
  bim(:,:,i) = imrotate(bim1, -(i-1)*20, 'crop');
end

% make pictures of positive weights bs adding up weighted glyphs
s = size(w);    
w(w < 0) = 0;    
im = zeros(bs*s(1), bs*s(2));
for i = 1:s(1),
  iis = (i-1)*bs+1:i*bs;
  for j = 1:s(2),
    jjs = (j-1)*bs+1:j*bs;          
    for k = 1:9,
      im(iis,jjs) = im(iis,jjs) + bim(:,:,k) * w(i,j,k+18);
    end
  end
end

scale = max(max(w(:)),max(-w(:)));
im = im / scale;
