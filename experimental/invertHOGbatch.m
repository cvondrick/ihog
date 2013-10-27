% invertHOGbatch(feats, pd)
%
% This function is a candidate to become main invertHOG(), but it is more
% complex.
function im = invertHOGbatch(feat, pd, verbose),

if ~exist('pd', 'var') || isempty(pd),
  global ihog_pd
  if isempty(ihog_pd),
    ihog_pd = load('pd.mat');
  end
  pd = ihog_pd;
end
if ~exist('verbose', 'var'),
  verbose = true;
end

start = tic();
if verbose, t = tic(); fprintf('ihog: batch: preprocess... '); end;

par = 5;
feat = padarray(feat, [par par 0 0], 0);

[ny, nx, ~, nn] = size(feat);

% pad feat with 0s if not big enough
if size(feat,1) < pd.ny,
  feat = padarray(feat, [pd.ny - size(x,1) 0 0 0], 0, 'post');
end
if size(feat,2) < pd.nx,
  feat = padarray(feat, [0 pd.nx - size(x,2) 0 0], 0, 'post');
end

% pad feat if dim lacks occlusion feature
if size(feat,3) == featuresdim()-1,
  feat(:, :, end+1, :) = 0;
end

if verbose, fprintf('%0.2fs sec\n', toc(t)); end

if verbose, t = tic(); fprintf('ihog: batch: extract... '); end;

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

if verbose, fprintf('%0.2fs sec\n', toc(t)); end;

if verbose, t = tic(); fprintf('ihog: batch: lasso... '); end;

% solve lasso problem
param.lambda = pd.lambda;
param.mode = 2;
param.pos = true;
a = full(mexLasso(single(windows), pd.dhog, param));
recon = pd.dgray * a;

if verbose, fprintf('%0.2fs sec\n', toc(t)); end;

if verbose, t = tic(); fprintf('ihog: batch: reconstruct... '); end;

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

if verbose, fprintf('%0.2fs sec\n', toc(t)); end;

if verbose, t = tic(); fprintf('ihog: batch: postprocess... '); end;

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

if verbose, fprintf('%0.2fs sec\n', toc(t)); end;

if verbose, fprintf('ihog: batch: inverted %i points of %ix%i in %0.2f sec\n', nn, ny-2*par, nx-2*par, toc(start)); end;
