% visualizeRotation(im, pd)
%
% Rotates im around and inverts each rotation. After a long processing time, it
% plays a movie of the inversion spinning.
function visualizeRotation(im, out, pd, angles),

if ~exist('angles', 'var'),
  angles = 0:3:360;
end

ny = (round(size(im,1)/pd.sbin))*pd.sbin+1;
nx = (round(size(im,2)/pd.sbin))*pd.sbin+1;

[~, maxdim] = max(size(im));
if maxdim == 1,
  im = padarray(im, [0 floor((size(im,1)-size(im,2))/2)], 0);
else,
  im = padarray(im, [floor((size(im,2)-size(im,1))/2) 0], 0);
end

if size(im,1) > size(im,2),
  im = padarray(im, [0 size(im,1)-size(im,2)], 0, 'post');
elseif size(im,1) < size(im,2),
  im = padarray(im, [size(im,2)-size(im,1) 0], 0, 'post');
end

fprintf('ihog: rotating: ');

for i=1:length(angles),
  fprintf('.');
  rim = imrotate(im, angles(i), 'bilinear', 'crop');
  feat = features(rim, 8);
  ihog = invertHOG(feat, pd);
  images{i} = ihog;
  subplot(121); imagesc(rim); axis image;
  subplot(122); imagesc(ihog); axis image;
  drawnow;

  imwrite(ihog, sprintf('%s/%i.jpg', out, i));
end
fprintf('\n');

clf;

while true,
  for i=1:length(angles),
    imagesc(images{i}); axis image;
    pause(0.25);
  end
end
