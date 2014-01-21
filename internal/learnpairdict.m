% learnpairdict(stream, n, k, size)
%
% This function learns a pair of dictionaries 'dgray' and 'dhog' to allow for
% regression between HOG and grayscale images.
%
% Arguments:
%   stream    List of filepaths where images are located
%   n         Number of window patches to extract in total
%   k         The size of the dictionary
%   dim       The size of the template patch to invert
%   lambda    Sparsity regularization parameter on alpha
%   iters     Number of iterations 
%   sbin      The HOG bin size
%   fast      If true, 'learn' a dictionary in real time (default false)
% 
% Returns a struct with fields:
%   dgray     A dictionary of gray elements
%   dhog      A dictionary of HOG elements

function pd = learnpairdict(stream, n, k, ny, nx, lambda, reggray, reghog, iters, sbin, fast),

if ~exist('n', 'var'),
   n = 1000000;
end
if ~exist('k', 'var'),
  k = 1024;
end
if ~exist('ny', 'var'),
  ny = 5;
end
if ~exist('nx', 'var'),
  nx = 5;
end
if ~exist('lambda', 'var'),
  lambda = 1;
end
if ~exist('reggray', 'var'),
  reggray = .001;
end
if ~exist('reghog', 'var'),
  reghog = .001;
end
if ~exist('iters', 'var'),
  iters = 1000;
end
if ~exist('sbin', 'var'),
  sbin = 8;
end
if ~exist('fast', 'var'),
  fast = false;
end

graysize = (ny+2)*(nx+2)*sbin^2;

t = tic;

stream = resolvestream(stream);
[data, trainims] = getdata(stream, n, [ny nx], sbin);

fprintf('ihog: compute dataset mean\n');
mu = mean(data, 2);
graymu = mu(1:graysize);
hogmu  = mu(graysize+1:end);

for i=1:size(data,2),
  data(:, i) = data(:, i) - mu;
end

fprintf('ihog: compute dataset covariance\n');
graysig = 1/(n-1) * data(1:graysize, :) * data(1:graysize, :)';
graysig = (graysig + graysig') / 2;

hogsig = 1/(n-1) * data(graysize+1:end, :) * data(graysize+1:end, :)';
hogsig = (hogsig + hogsig') / 2;

fprintf('ihog: compute whitening matrix\n');
[wgray, cgray] = whiten(graysig, reggray);
[whog,  chog]  = whiten(hogsig, reghog);

fprintf('ihog: whitening data\n');
data = blkdiag(wgray, whog) * data;

if fast,
  dict = pickrandom(data, k); 
else,
  dict = lasso(data, k, iters, lambda);
end

pd.dgray = dict(1:graysize, :);
pd.dhog = dict(graysize+1:end, :);
pd.n = n;
pd.k = k;
pd.ny = ny;
pd.nx = nx;
pd.sbin = sbin;
pd.iters = iters;
pd.lambda = lambda;
pd.trainims = trainims;
pd.wgray = wgray;
pd.cgray = cgray;
pd.whog = whog;
pd.chog = chog;
pd.mugray = graymu;
pd.muhog = hogmu;

fprintf('ihog: paired dictionaries learned in %0.3fs\n', toc(t));



% lasso(data)
%
% Learns the pair of dictionaries for the data terms.
function dict = lasso(data, k, iters, lambda),

param.K = k;
param.lambda = lambda;
param.mode = 0;
param.modeD = 0;
param.iter = 100;
param.numThreads = 12;
param.verbose = 1;
param.batchsize = 400;

fprintf('ihog: lasso\n');
model = struct();
for i=1:(iters/param.iter),
  fprintf('ihog: lasso: master iteration #%i\n', i);
  [dict, model] = mexTrainDL(data, param, model);
  model.iter = i*param.iter;
  param.D = dict;
end



% calculate whitening and coloring
function [w, c] = whiten(sig, r),
[v,d] = eig(sig);
d = diag(d);
d(d < r) = r;
w = v * diag(1./sqrt(d)) * v';
c = v * diag(sqrt(d)) * v';



% pickrandom(data, k)
%
% Picks a random 'k' elements from 'data' for fast training. This option is mostly
% useful for debugging purposes. The learned dictionary is usually always better than
% this mode. But, this method still produces surprisingly good reconstructions.
function dict = pickrandom(data, k),

fprintf('ihog: sampling %i random elements for dictionary instead of learning\n', k)
order = randperm(size(data, 2));
order = order(1:k);
dict = data(:, order);



% getdata(stream, n, dim, sbin)
%
% Reads in the stream and extracts windows along with their HOG features.
function [data, images] = getdata(stream, n, dim, sbin),

ny = dim(1);
nx = dim(2);

fprintf('ihog: allocating data store: %.02fGB\n', ...
        ((ny+2)*(nx+2)*sbin^2+ny*nx*featuresdim())*n*4/1024/1024/1024);
data = zeros((ny+2)*(nx+2)*sbin^2+ny*nx*featuresdim(), n, 'single');
c = 1;

fprintf('ihog: loading data: ');
while true,
  for k=1:length(stream),
    fprintf('.');

    im = double(imread(stream{k})) / 255.;
    im = mean(im,3);
    feat = features(repmat(im, [1 1 3]), sbin);

    for i=1:size(feat,1) - dim(1),
      for j=1:size(feat,2) - dim(2),
        if n <= 100000 && rand() > 0.1,
          continue;
        end

        featpoint = feat(i:i+ny-1, ...
                         j:j+ny-1, :);
        graypoint = im((i-1)*sbin+1:(i+1+ny)*sbin, ...
                       (j-1)*sbin+1:(j+1+nx)*sbin);
        data(:, c) = single([graypoint(:); featpoint(:)]);

        c = c + 1;
        if c >= n,
          images = stream(1:k);
          fprintf('\n');
          fprintf('ihog: loaded %i windows\n', c);
          return;
        end
      end
    end
  end
  fprintf('\n');
  fprintf('ihog: warning: wrapping around dataset!\n');
end
