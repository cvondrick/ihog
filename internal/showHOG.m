% Visualize HOG features/weights.
%   visualizeHOG(w)
function showHOG(w)

% Make pictures of positive and negative weights
bs = 20;
%w = w(:,:,19:28);
scale = max(max(w(:)),max(-w(:)));
pos = HOGpicture(w, bs) * 255/scale;
neg = HOGpicture(-w, bs) * 255/scale;

% Put pictures together and draw
buff = 10;
pos = padarray(pos, [buff buff], 128, 'both');
if min(w(:)) < 0
  neg = padarray(neg, [buff buff], 128, 'both');
  im = uint8([pos; neg]);
else
  im = uint8(pos);
end
imagesc(im); 
colormap gray;
axis image;


% Make picture of positive HOG weights.
%   im = HOGpicture(w, bs)
%
% Written by Amir Rosenfeld
function im = HOGpicture(w, bs)

if ~exist('bs','var')
  bs = 20;
end

% construct a "glyph" for each orientaion
s = size(w);
w(w<0) = 0;
bim1 = zeros(bs, bs);
bim1(:,round(bs/2):round(bs/2)+1) = 1;
bim = zeros([size(bim1) 9]);
bim(:,:,1) = bim1;
for i = 2:9,
  bim(:,:,i) = imrotate(bim1, -(i-1)*20, 'crop');
end

bim = cat(3,bim,bim);
bim_ = reshape(bim,[],18);

B = im2col( zeros(bs*s(1), bs*s(2)),[bs bs],'distinct');
w_ = reshape(w(:,:,1:18),[],18);
im = col2im(bim_*w_',[bs bs],[bs*s(1), bs*s(2)],'distinct');
