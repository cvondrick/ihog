function [im, prev] = invertCNN(feat, pd, prev),

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

if ~exist('prev', 'var'),
  prev.a = zeros(0, 0, 0);
end
if ~isfield(prev, 'gam'),
  prev.gam = 10;
end
if ~isfield(prev, 'sig'),
  prev.sig = 1;
end
if ~isfield(prev, 'mode'),
  prev.mode = 'xpass';
end
prevnum = size(prev.a, 3);
prevnuma = size(prev.a, 2);

windows = zeros(prod(pd.featdim), size(feat,2));
for i=1:size(feat,2),
  elem = feat(:, i);
  elem(:) = elem(:) - mean(elem(:));
  elem(:) = elem(:) / (sqrt(sum(elem(:).^2) + eps));
  windows(:, i) = elem(:);
end

% incorporate constraints for multiple inversions
dcnn = pd.dcnn;
mask = logical(ones(size(windows)));
if prevnum > 0,
  windows = padarray(windows, [prevnum*prevnuma 0], 0, 'post');
  mask = cat(1, mask, repmat(logical(eye(prevnuma, size(windows,2))), [prevnum 1]));
  offset = size(dcnn, 1);
  dcnn = padarray(dcnn, [prevnum*prevnuma 0], 0, 'post');

  if strcmp(prev.mode, 'xpass'),
    % build blurred dictionary
    dblur = xpassdict(pd.drgb, pd.imdim, prev.sig);
    D = dblur' * dblur;
  elseif strcmp(prev.mode, 'rgb'),
    selector = ones(1, pd.imdim(1) * pd.imdim(2)) / (pd.imdim(1) * pd.imdim(2));
    colortrans = blkdiag(selector, selector, selector);
    D = pd.drgb' * colortrans' * colortrans * pd.drgb;
  end

  for i=1:prevnum,
    dcnn(offset+(i-1)*prevnuma+1:offset+i*prevnuma, :) = sqrt(prev.gam) * prev.a(:, :, i)' * D;
  end
end

% solve lasso problem
param.lambda = pd.lambda * size(windows,1) / (prod(pd.featdim) + prevnum);
param.mode = 2;
param.pos = true;
a = full(mexLassoMask(single(windows), dcnn, mask, param));
recon = pd.drgb * a;

%fprintf('icnn: sparsity=%0.2f\n', sum(a(:) == 0) / length(a(:)));

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
