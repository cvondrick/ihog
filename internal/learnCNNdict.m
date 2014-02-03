function pd = learnCNNdict(chunkmasterfile, k, lambda, iters),

if ~exist('k', 'var'),
  k = 1024;
end
if ~exist('lambda', 'var'),
  lambda = 0.02; % 0.02 is best so far
end
if ~exist('iters', 'var'),
  iters = 10;
end

t = tic;

fprintf('icnn: locating chunkfiles...\n');
master = load(chunkmasterfile);
master = master.master;

chunks = cell(length(master.files), 1);
fprintf('icnn: %i valid chunks\n', length(master.files));
for i=1:length(chunks),
  chunks{i} = sprintf('%s/%s', fileparts(chunkmasterfile), master.files{i});
  fprintf('icnn: chunk %i: %s\n', i, chunks{i})
end

param.K = k;
param.lambda = lambda;
param.mode = 2;
param.modeD = 0;
param.iter = 100;
param.numThreads = 12;
param.verbose = 1;
param.batchsize = 400;
param.posAlpha = true;

model = struct();

chunkid = 1;
for i=1:iters,
  fprintf('icnn: master iteration #%i of %i\n', i, iters);

  fprintf('icnn: loading chunk %i: %s\n', chunkid, chunks{chunkid});
  data = load(chunks{chunkid});
  fprintf('icnn: chunk %i has d=%i, n=%i\n', chunkid, size(data.data, 1), size(data.data,2));

  [dict, model] = mexTrainDL(data.data, param, model);
  model.iter = i*param.iter;
  param.D = dict;

  chunkid = mod(chunkid, length(chunks))+1;
end

pd.drgb = dict(1:prod(master.imdim), :);
pd.dcnn = dict(prod(master.imdim)+1:end, :);
pd.n = master.n;
pd.k = k;
pd.imdim = master.imdim;
pd.featdim = master.featdim;
pd.chunkmaster = master;
pd.lambda = lambda;
pd.feat = 'CNN';
pd.iters = iters;

fprintf('icnn: paired dictionaries learned in %0.3fs\n', toc(t));
