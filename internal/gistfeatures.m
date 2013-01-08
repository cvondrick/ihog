function gist = gistfeatures(im),

param.imageSize = 128;
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

if ~exist('im', 'var'),
  gist = 512;
  return;
end

gist = LMgist(im, '', param);
