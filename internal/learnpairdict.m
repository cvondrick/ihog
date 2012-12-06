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
%   gamma     Dictionary L2 regularization parameter
%   iters     Number of iterations 
%   sbin      The HOG bin size
% 
% Returns a struct with fields:
%   dgray     A dictionary of gray elements
%   dhog      A dictionary of HOG elements

function pd = learnpairdict(stream, n, k, ny, nx, lambda, gamma, iters, sbin),

if ~exist('n', 'var'),
   n = 100000;
end
if ~exist('k', 'var'),
  k = 100;
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
if ~exist('gamma', 'var'),
  gamma = 0.05;
end
if ~exist('iters', 'var'),
  iters = 2000;
end
if ~exist('sbin', 'var'),
  sbin = 8;
end

graysize = (ny+2)*(nx+2)*sbin^2;

t = tic;

stream = resolvestream(stream);
data = getdata(stream, n, [ny nx], sbin);

data(1:graysize, :) = whiten(data(1:graysize, :));
data(graysize+1:end, :) = whiten(data(graysize+1:end, :));

dict = lasso(data, k, iters, lambda, gamma);

pd.dgray = dict(1:graysize, :);
pd.dhog = dict(graysize+1:end, :);
pd.n = n;
pd.k = k;
pd.ny = ny;
pd.nx = nx;
pd.sbin = sbin;
pd.iters = iters;
pd.lambda = lambda;
pd.gamma = gamma;

fprintf('ihog: paired dictionaries learned in %0.3fs\n', toc(t));



% lasso(data)
%
% Learns the pair of dictionaries for the data terms.
function dict = lasso(data, k, iters, lambda, gamma),

param.K = k;
param.lambda = lambda;
param.mode = 2;
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



% whiten(in)
%
% Whitens the input feature with zero mean and unit variance
function data = whiten(data),
fprintf('ihog: whiten: zero mean\n');
mu = mean(data(:));
for i=1:size(data,2),
  data(:, i) = data(:, i) - mean(data(:, i));
end
fprintf('ihog: whiten: unit variance\n');
for i=1:size(data,2),
  data(:, i) = data(:, i) / (std(data(:, i)) + 1);
end



% getdata(stream, n, dim, sbin)
%
% Reads in the stream and extracts windows along with their HOG features.
function data = getdata(stream, n, dim, sbin),

ny = dim(1);
nx = dim(2);

fprintf('ihog: allocating data store: %.02fGB\n', ...
        ((ny+2)*(nx+2)*sbin^2+ny*nx*32)*n*4/1024/1024/1024);
data = zeros((ny+2)*(nx+2)*sbin^2+ny*nx*32, n, 'single');
c = 1;

fprintf('ihog: loading data: ');
while true,
  for i=1:length(stream),
    fprintf('.');
    im = double(imread(stream{i})) / 255.;
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



% resolvestream(stream)
%
% If stream is a directory, convert to list of paths. Otherwise,
% do nothing.
function stream = resolvestream(stream),

if isstr(stream),
  fprintf('ihog: reading images from directory: %s\n', stream);
  directory = stream;
  files = dir(stream);
  clear stream;
  c = 1;
  for i=1:length(files);
    if ~files(i).isdir,
      stream{c} = [directory '/' files(i).name];
      c = c + 1;
    end
  end
  fprintf('ihog: stream resolved to %i images\n', length(stream));
end
