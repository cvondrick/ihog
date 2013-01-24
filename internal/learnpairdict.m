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
%   dim       The gist dimension
%   fast      If true, 'learn' a dictionary in real time (default false)
% 
% Returns a struct with fields:
%   dgray     A dictionary of gray elements
%   dhog      A dictionary of HOG elements

function pd = learnpairdict(stream, n, k, ny, nx, lambda, iters, dim, fast),

if ~exist('n', 'var'),
   n = 100000;
end
if ~exist('k', 'var'),
  k = 1000;
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
if ~exist('dim', 'var'),
  dim = 128;
end
if ~exist('fast', 'var'),
  fast = false;
end

graysize = dim^2;

t = tic;

stream = resolvestream(stream);
[data, trainims] = getdata(stream, n, [ny nx], dim);

fprintf('ihog: normalize\n');
for i=1:size(data,2),
  data(1:graysize, i) = data(1:graysize, i) - mean(data(1:graysize, i));
  data(1:graysize, i) = data(1:graysize, i) / (sqrt(sum(data(1:graysize, i).^2) + 1));
  data(graysize+1:end, i) = data(graysize+1:end, i) - mean(data(graysize+1:end, i));
  data(graysize+1:end, i) = data(graysize+1:end, i) / (sqrt(sum(data(graysize+1:end, i).^2) + 1));
end

if fast,
  dict = pickrandom(data, k); 
else,
  dict = lasso(data, k, iters, lambda);
end

pd.dgray = dict(1:graysize, :);
pd.dhog = dict(graysize+1:end, :);
pd.n = size(data,2);
pd.k = k;
pd.ny = ny;
pd.nx = nx;
pd.dim = dim;
pd.iters = iters;
pd.lambda = lambda;
pd.trainims = trainims;

fprintf('igist: paired dictionaries learned in %0.3fs\n', toc(t));



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

fprintf('igist: lasso\n');
model = struct();
for i=1:(iters/param.iter),
  fprintf('igist: lasso: master iteration #%i\n', i);
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



% getdata(stream, n, dim, dim)
%
% Reads in the stream and extracts windows along with their HOG features.
function [data, images] = getdata(stream, n, dim, gdim), 

ny = dim(1);
nx = dim(2);

if n == -1,
  n = length(stream);
  fprintf('igist: setting n to number of images: %i\n', n);
end

fprintf('igist: allocating data store: %.02fGB\n', ...
        (gdim^2+gistfeatures)*n*4/1024/1024/1024);
data = zeros(gdim^2+gistfeatures, n, 'single');
c = 1;

skipby = 32;

fprintf('igist: loading data: ');
while true,
  for k=1:length(stream),
    fprintf('.');

    datum = load(stream{k});

    for x=1:length(datum.data),
      data(:, c) = datum.data{x};
      c = c + 1;
      if c >= n,
        images = stream(1:k);
        fprintf('\n');
        fprintf('igist: loaded %i patches\n', c);
        return;
      end
    end
  end

  fprintf('\n');
  fprintf('igist: warning: wrapping around dataset!\n');
end




% resolvestream(stream)
%
% If stream is a directory, convert to list of paths. Otherwise,
% do nothing.
function stream = resolvestream(stream),

if isstr(stream),
  fprintf('igist: reading gist files from directory: %s\n', stream);
  directory = stream;
  files = dir(stream);
  clear stream;
  c = 1;
  iii = randperm(length(files));
  for i=1:length(files);
    if ~files(iii(i)).isdir,
      stream{c} = [directory '/' files(iii(i)).name];
      c = c + 1;
    end
  end
  fprintf('igist: stream resolved to %i images\n', length(stream));
end
