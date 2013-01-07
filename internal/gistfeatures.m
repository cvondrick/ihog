function gist = gistfeatures(im),

param.imageSize = 128;
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

gist = LMgist(im, '', param);
