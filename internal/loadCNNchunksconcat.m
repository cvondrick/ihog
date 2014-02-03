function concat = loadCNNchunksconcat(chunkmasterfile, num, rgb, feat),

[master, chunks] = resolveCNNchunks(chunkmasterfile);

if ~exist('num', 'var') || num < 0,
  num = master.n;
end
if ~exist('rgb', 'var'),
  rbg = true;
end
if ~exist('feat', 'var'),
  feat = true;
end

elemsize = rgb*prod(master.imdim) + feat*prod(master.featdim);
fprintf('icnn: allocating storage: %0.2fGB\n', elemsize * num * 4 / 1024^3);
concat = zeros(elemsize, num, 'single');

iii = logical(zeros(prod(master.imdim)+prod(master.featdim), 1));
if rgb,
  iii(1:prod(master.imdim)) = 1;
end
if feat,
  iii(prod(master.imdim)+1:end) = 1;
end

c = 1;
for i=1:length(chunks),
  fprintf('icnn: loading %s\n', chunks{i});
  data = load(chunks{i});

  take = min(size(data.data,2), num-c+1);
  concat(:, c:c+take-1) = data.data(iii, randperm(size(data.data,2), take));

  c = c + take;
  fprintf('icnn: loaded %i of %i\n', c-1, num);

  if c >= num,
    break;
  end
end
