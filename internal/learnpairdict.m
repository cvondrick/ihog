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
%   fast      If true, 'learn' a dictionary in real time (default false)
% 
% Returns a struct with fields:
%   dgray     A dictionary of gray elements
%   dhog      A dictionary of HOG elements
function pd = learnpairdict(data, k, ny, nx, lambda, iters, fast),

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
  lambda = 0.02; % 0.02 is best so far
end
if ~exist('iters', 'var'),
  iters = 1000;
end
if ~exist('fast', 'var'),
  fast = false;
end

n = size(data,1);
graysize = ny*nx*3;

t = tic;

fprintf('ihog: normalize\n');
for i=1:size(data,2),
  data(1:graysize, i) = data(1:graysize, i) - mean(data(1:graysize, i));
  data(1:graysize, i) = data(1:graysize, i) / (sqrt(sum(data(1:graysize, i).^2) + eps));
  data(graysize+1:end, i) = data(graysize+1:end, i) - mean(data(graysize+1:end, i));
  data(graysize+1:end, i) = data(graysize+1:end, i) / (sqrt(sum(data(graysize+1:end, i).^2) + eps));
end

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
pd.iters = iters;
pd.lambda = lambda;

fprintf('ihog: paired dictionaries learned in %0.3fs\n', toc(t));



% lasso(data)
%
% Learns the pair of dictionaries for the data terms.
function dict = lasso(data, k, iters, lambda),

param.K = k;
param.lambda = lambda;
param.mode = 2;
param.modeD = 0;
param.iter = 100;
param.numThreads = 12;
param.verbose = 1;
param.batchsize = 400;
param.posAlpha = true;

fprintf('ihog: lasso\n');
model = struct();
for i=1:(iters/param.iter),
  fprintf('ihog: lasso: master iteration #%i\n', i);
  [dict, model] = mexTrainDL(data, param, model);
  model.iter = i*param.iter;
  param.D = dict;
end



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
