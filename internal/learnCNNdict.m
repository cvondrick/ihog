% learnCNNdict(stream, n, k, size)
function pd = learnCNNdict(datafile, k, lambda, iters, fast),

if ~exist('k', 'var'),
  k = 1024;
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

t = tic;

fprintf('icnn: loading data...\n');
load(datafile); % loads: data, imdim, featdim

n = size(data,2);
rgbsize = prod(imdim);

fprintf('icnn: graydim=%i, featdim=%i, n=%i, k=%i\n', rgbsize, size(data,1)-rgbsize, n, k);

fprintf('icnn: normalize\n');
for i=1:size(data,2),
  data(1:rgbsize, i) = data(1:rgbsize, i) - mean(data(1:rgbsize, i));
  data(1:rgbsize, i) = data(1:rgbsize, i) / (sqrt(sum(data(1:rgbsize, i).^2) + eps));
  data(rgbsize+1:end, i) = data(rgbsize+1:end, i) - mean(data(rgbsize+1:end, i));
  data(rgbsize+1:end, i) = data(rgbsize+1:end, i) / (sqrt(sum(data(rgbsize+1:end, i).^2) + eps));
end

if fast,
  dict = pickrandom(data, k); 
else,
  dict = lasso(data, k, iters, lambda);
end

pd.drgb = dict(1:rgbsize, :);
pd.dcnn = dict(rgbsize+1:end, :);
pd.n = n;
pd.k = k;
pd.imdim = imdim;
pd.iters = iters;
pd.lambda = lambda;
pd.feat = 'CNN';

fprintf('icnn: paired dictionaries learned in %0.3fs\n', toc(t));



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

fprintf('icnn: lasso\n');
model = struct();
for i=1:(iters/param.iter),
  fprintf('icnn: lasso: master iteration #%i\n', i);
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

fprintf('icnn: sampling %i random elements for dictionary instead of learning\n', k)
order = randperm(size(data, 2));
order = order(1:k);
dict = data(:, order);
