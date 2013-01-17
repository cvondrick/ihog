% visualizeRotation(im, pd)
%
% Rotates im around and inverts each rotation. After a long processing time, it
% plays a movie of the inversion spinning.
function visualizeRotation(im, pd, angles),

if ~exist('angles', 'var'),
  angles = 0:3:360;
end

ny = (round(size(im,1)/pd.sbin))*pd.sbin+1;
nx = (round(size(im,2)/pd.sbin))*pd.sbin+1;

fprintf('ihog: rotating: ');

for i=1:length(angles),
  fprintf('.');
  rim = imrotate(im, angles(i), 'bilinear');
  feat = features(rim, 8);
  ihog = invertHOG(feat, pd);
  images{i} = ihog;
  subplot(121); imagesc(rim); axis image;
  subplot(122); imagesc(ihog); axis image;
  drawnow;
end
fprintf('\n');

clf;

while true,
  for i=1:length(angles),
    imagesc(images{i}); axis image;
    pause(0.25);
  end
end
