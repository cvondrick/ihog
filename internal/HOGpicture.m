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

end
