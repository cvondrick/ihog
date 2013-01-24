function out = visualizeGist(gist),

param.imageSize = 128;
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;
param.boundaryExtension = 32; % number of pixels to pad
param.G = createGabor(param.orientationsPerScale, param.imageSize+2*param.boundaryExtension);

out = im2double(showGist(gist, param));
