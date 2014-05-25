function [im, prev] = invertCNN(feat, pd, prev, w),

% load dictionary if missing
if ~exist('pd', 'var') || isempty(pd),
  global icnn_pd
  if isempty(icnn_pd),
    if ~exist('pd-icnn.mat', 'file'),
      error('please create pd-icnn.mat or otherwise specify pd');
    end
    icnn_pd = load('pd-icnn.mat');
  end
  pd = icnn_pd;
end

% build previous data structure
if ~exist('prev', 'var'),
  prev = struct();
end
if ~isfield(prev, 'a'),
  prev.a = zeros(0, 0, 0);
end
if ~isfield(prev, 'mode'),
  prev.mode = 'rgb';
end
if ~isfield(prev, 'gam'),
  prev.gam = 1;
end
if ~isfield(prev, 'sig'),
  prev.sig = 1;
end
if ~isfield(prev, 'slices'),
  prev.slices = 2;
end
prevnum = size(prev.a, 3);
prevnuma = size(prev.a, 2);

% check for model weights
if ~exist('w', 'var'),
  w = ones(pd.featdim, 1);
end

% do some error checking
if size(feat,1) ~= pd.featdim,
  error(sprintf('expected feat to be %ixK, instead got %ix%i', pd.featdim, size(feat,1), size(feat,2)));
end
if size(w,1) ~= pd.featdim || size(w,2) ~= 1,
  error(sprintf('expected w to be %ix1, instead got %ix%i', pd.featdim, size(w,1), size(w,2)));
end

t = tic();

% process windows
windows = zeros(prod(pd.featdim), size(feat,2));
for i=1:size(feat,2),
  elem = feat(:, i);
  elem(:) = elem(:) - mean(elem(:));
  elem(:) = elem(:) / (sqrt(sum(elem(:).^2) + eps));
  windows(:, i) = elem(:);
end

% we will modify this variable later, so keep backup
origwindows = windows;

% copy dcnn since we'll manipulate it now
dcnn = pd.dcnn;

% incorporate model weights, if any
if any(w ~= 1),
  w = abs(w);
  w = w / norm(w) * sqrt(pd.featdim);
  w = diag(w);
  dcnn = w * dcnn;
  windows = w * windows;
end

% incorporate constraints for multiple inversions
mask = logical(ones(size(windows)));
offset = size(dcnn, 1);
if prevnum > 0,
  windows = padarray(windows, [prevnum*prevnuma 0], 0, 'post');
  mask = cat(1, mask, repmat(logical(eye(prevnuma, size(windows,2))), [prevnum 1]));
  dcnn = padarray(dcnn, [prevnum*prevnuma 0], 0, 'post');

  if strcmp(prev.mode, 'standard'),
    D = pd.drgb' * pd.drgb;

  elseif strcmp(prev.mode, 'xpass'),
    % build blurred dictionary
    dblur = xpassdict(pd.drgb, pd.imdim, prev.sig);
    D = dblur' * dblur;

  elseif strcmp(prev.mode, 'rgb'),
    % build selector tensor 
    selector = zeros(prev.slices^2, pd.imdim(1), pd.imdim(2));

    fil = 1;
    %fil = fspecial('gaussian', [pd.imdim(1)/prev.slices pd.imdim(2)/prev.slices], prev.sig);

    for i=1:prev.slices,
      iii = (i-1)*pd.imdim(1)/prev.slices+1 : i*pd.imdim(1)/prev.slices;
      for j=1:prev.slices,
        jjj = (j-1)*pd.imdim(2)/prev.slices+1 : j*pd.imdim(2)/prev.slices;
        selector((i-1)*prev.slices+j, iii, jjj) = fil;
      end
    end

    %selector = selector / (pd.imdim(1) * pd.imdim(2) / prev.slices^2);
    selector = reshape(selector, [prev.slices^2 pd.imdim(1)*pd.imdim(2)]);
    colortrans = blkdiag(selector, selector, selector);

    D = pd.drgb' * colortrans' * colortrans * pd.drgb;

  elseif strcmp(prev.mode, 'hog-dets'),
    D = pd.dhog' * prev.dets * prev.dets' * pd.dhog;

  elseif strcmp(prev.mode, 'hog-metric'),
    D = pd.dhog' * prev.metric * pd.dhog;

  elseif strcmp(prev.mode, 'hog'),
    D = pd.dhog' * pd.dhog;
  end

  for i=1:prevnum,
    dcnn(offset+(i-1)*prevnuma+1:offset+i*prevnuma, :) = sqrt(prev.gam) * prev.a(:, :, i)' * D;
  end
end

% solve lasso problem
param.lambda = pd.lambda * size(windows,1) / (prod(pd.featdim) + prevnum);
param.mode = 2;
param.pos = true;
%param.L = 20;
a = full(mexLassoMask(single(windows), dcnn, mask, param));
recon = pd.drgb * a;

% post process the result
im = reshape(recon, [pd.imdim size(windows,2)]);
for i=1:size(feat,2),
  img = im(:, :, :, i);
  img(:) = img(:) - min(img(:));
  img(:) = img(:) / max(img(:));
  im(:, :, :, i) = img;
end

% build previous information
if prevnum > 0,
  prev.a = cat(3, prev.a, a);
else,
  prev.a = a;
end

% output some debugging information
featcost = pd.dcnn * a - origwindows;
featcost = norm(featcost(:));
diversitycost = dcnn(offset+1:end, :) * a;
diversitycost = norm(diversitycost(:));
fprintf('icnn: finished in %0.2fs\n', toc(t));
fprintf('icnn:   sparsity = %i or %0.2f\n', sum(a(:) ~= 0), sum(a(:) == 0) / length(a(:)));
fprintf('icnn:   feat cost = %f\n', featcost);
fprintf('icnn:   diversity cost = %f\n', diversitycost);
