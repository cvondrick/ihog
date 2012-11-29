% Visualize HOG features/weights.
%   visualizeHOG(w)
function showHOG(w)

% Make pictures of positive and negative weights
bs = 20;
w = w(:,:,19:27);
scale = max(max(w(:)),max(-w(:)));
pos = HOGpicture(w, bs) * 255/scale;
neg = HOGpicture(-w, bs) * 255/scale;

% Put pictures together and draw
buff = 10;
if min(w(:)) < 0
  pos = padarray(pos, [buff buff], 128, 'both');
  neg = padarray(neg, [buff buff], 128, 'both');
  im = uint8([pos neg]);
else
  im = uint8(pos);
end
imagesc(im); 
colormap gray;
axis image;




% Make picture of positive HOG weights.
%   im = HOGpicture(w, bs)
function im = HOGpicture(w, bs)

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
      im(iis,jjs) = im(iis,jjs) + bim(:,:,k) * w(i,j,k);
    end
  end
end
