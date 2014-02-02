% invertHOG(feat)
%
% This function recovers the natural image that may have generated the HOG
% feature 'feat'. Usage is simple:
%
%   >> feat = features(im, 8);
%   >> ihog = invertHOG(feat);
%   >> imagesc(ihog); axis image;
%
% By default, invertHOG() will load a prelearned paired dictionary to perform
% the inversion. However, if you want to pass your own, you can specify the
% optional second parameter to use your own parameters:
% 
%   >> pd = learnpairdict('/path/to/images');
%   >> ihog = invertHOG(feat, pd);
%
% This function should take no longer than a second to invert any reasonably sized
% HOG feature point on a 12 core machine.
%
% If you have many points you wish to invert, this function can be vectorized.
% If 'feat' is size AxBxCxK, then it will invert K HOG features each of size
% AxBxC. It will return an PxQxK image tensor where the last channel is the kth
% inversion. This is usually significantly faster than calling invertHOG() 
% multiple times.
function im = invertHOG(feat, pd),

if ~exist('pd', 'var'),
  global ihog_pd
  if isempty(ihog_pd),
    if ~exist('pd.mat', 'file'),
      fprintf('ihog: notice: unable to find paired dictionary\n');
      fprintf('ihog: notice: attempting to download in 3');
      pause(1); fprintf('\b2'); pause(1); fprintf('\b1'); pause(1);
      fprintf('\b0\n');
      fprintf('ihog: notice: downloading...');
      urlwrite('http://people.csail.mit.edu/vondrick/pd.mat', 'pd.mat');
      fprintf('done\n');
    end
    ihog_pd = load('pd.mat');
  end
  pd = ihog_pd;
end

par = 5;
feat = padarray(feat, [par par 0 0], 0);

[ny, nx, ~, nn] = size(feat);

% pad feat if dim lacks occlusion feature
if size(feat,3) == featuresdim()-1,
  feat(:, :, end+1, :) = 0;
end

% extract every window 
windows = zeros(pd.ny*pd.nx*featuresdim(), (ny-pd.ny+1)*(nx-pd.nx+1)*nn);
c = 1;
for k=1:nn,
  for i=1:size(feat,1) - pd.ny + 1,
    for j=1:size(feat,2) - pd.nx + 1,
      hog = feat(i:i+pd.ny-1, j:j+pd.nx-1, :, k);
      hog = hog(:) - mean(hog(:));
      hog = hog(:) / sqrt(sum(hog(:).^2) + eps);
      windows(:,c)  = hog(:);
      c = c + 1;
    end
  end
end

% solve lasso problem
param.lambda = pd.lambda;
param.mode = 2;
param.pos = true;
a = full(mexLasso(single(windows), pd.dhog, param));
recon = pd.dgray * a;

% reconstruct
fil     = fspecial('gaussian', [(pd.ny+2)*pd.sbin (pd.nx+2)*pd.sbin], 9);
im      = zeros((size(feat,1)+2)*pd.sbin, (size(feat,2)+2)*pd.sbin, nn);
weights = zeros((size(feat,1)+2)*pd.sbin, (size(feat,2)+2)*pd.sbin, nn);
c = 1;
for k=1:nn,
  for i=1:size(feat,1) - pd.ny + 1,
    for j=1:size(feat,2) - pd.nx + 1,
      patch = reshape(recon(:, c), [(pd.ny+2)*pd.sbin (pd.nx+2)*pd.sbin]);
      patch = patch .* fil;

      iii = (i-1)*pd.sbin+1:(i-1)*pd.sbin+(pd.ny+2)*pd.sbin;
      jjj = (j-1)*pd.sbin+1:(j-1)*pd.sbin+(pd.nx+2)*pd.sbin;

      im(iii, jjj, k) = im(iii, jjj, k) + patch;
      weights(iii, jjj, k) = weights(iii, jjj, k) + 1;

      c = c + 1;
    end
  end
end

% post processing averaging and clipping
im = im ./ weights;
im = im(1:(ny+2)*pd.sbin, 1:(nx+2)*pd.sbin, :);
for k=1:nn,
  img = im(:, :, k);
  img(:) = img(:) - min(img(:));
  img(:) = img(:) / max(img(:));
  im(:, :, k) = img;
end

im = im(par*pd.sbin:end-par*pd.sbin-1, par*pd.sbin:end-par*pd.sbin-1, :);
