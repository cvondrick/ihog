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
% 
% Returns a struct with fields:
%   dgray     A dictionary of gray elements
%   dhog      A dictionary of HOG elements

function pd = learnpairdict(stream, n, k, ny, nx, lambda, iters, dim),

if ~exist('n', 'var'),
   n = -1;
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
  iters = 2000;
end
if ~exist('dim', 'var'),
  dim = 128;
end

graysize = dim^2;

t = tic;

stream = resolvestream(stream);
data = getdata(stream, n, [ny nx], dim);

data(1:graysize, :) = whiten(data(1:graysize, :));
data(graysize+1:end, :) = whiten(data(graysize+1:end, :));

dict = lasso(data, k, iters, lambda);

pd.dgray = dict(1:graysize, :);
pd.dhog = dict(graysize+1:end, :);
pd.n = size(data,2);
pd.k = k;
pd.ny = ny;
pd.nx = nx;
pd.dim = dim;
pd.iters = iters;
pd.lambda = lambda;

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



% whiten(in)
%
% Whitens the input feature with zero mean and unit variance
function data = whiten(data),
fprintf('igist: whiten: zero mean\n');
for i=1:size(data,2),
  data(:, i) = data(:, i) - mean(data(:, i));
end
fprintf('igist: whiten: unit variance\n');
for i=1:size(data,2),
  data(:, i) = data(:, i) / (sqrt(sum(data(:, i).^2) + 1));
end



% getdata(stream, n, dim, dim)
%
% Reads in the stream and extracts windows along with their HOG features.
function data = getdata(stream, n, dim, gdim),

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

fprintf('igist: loading data: ');
while true,
  for i=1:length(stream),
    fprintf('.');
    im = double(imread(stream{i})) / 255.;
    im = imresizecrop(im, gdim);
    im = mean(im,3);
    feat = gistfeatures(repmat(im, [1 1 3]));

    data(:, c) = single([im(:); feat(:)]);

    c = c + 1;
    if c >= n,
      fprintf('\n');
      fprintf('igist: loaded %i images\n', c);
      return;
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
  fprintf('igist: reading images from directory: %s\n', stream);
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
