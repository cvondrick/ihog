function buildCNNchunks(sourcedir, outdir, chunksize),

if ~exist('chunksize', 'var'),
  chunksize = 100000;
end
if ~exist('layer', 'var'),
  layer = 'pool5_cudanet_out';
end
if ~exist('featdim', 'var'),
  featdim = [6 6 256];
end
if ~exist('imagedim', 'var'),
  imagedim = 64;
end
if ~exist('imagedimorig', 'var'),
  imagedimorig = 256;
end

imagedimorig = 256;
data = zeros(prod(featdim)+imagedim^2*3, chunksize, 'single');
c = 1;
chunkid = 1;
n = 0;

master.imdim = [imagedim imagedim 3];
master.featdim = featdim;

files = dir(sourcedir);
files = files(randperm(length(files)));
for i=1:length(files),
  if files(i).isdir,
    continue;
  end

  fprintf('icnn: load: %s (chunk has %i of %i)\n', files(i).name, c, chunksize);

  payload = load([sourcedir '/' files(i).name]);
  payloadfeatures = getfield(payload, layer);

  num = size(payloadfeatures,1);
  n = n + num;

  for j=1:num,
    im = squeeze(payload.images(j, :, :, :));
    im = imresize(im, [imagedim imagedim]);
    im = single(im) / 255.;

    feat = payloadfeatures(j, :, :, :);

    % normalize
    im(:) = im(:) - mean(im(:));
    im(:) = im(:) / (sqrt(sum(im(:).^2) + eps));

    feat(:) = feat(:) - mean(feat(:));
    feat(:) = feat(:) / (sqrt(sum(feat(:).^2) + eps));

    data(1:imagedim^2*3, c) = im(:);
    data(imagedim^2*3+1:end, c) = feat(:);

    c = c + 1;

    if c > chunksize,
      outpath = sprintf('%s/chunk-%i.mat', outdir, chunkid);
      fprintf('icnn: flush chunk %i to %s\n', chunkid, outpath);
      master.files{chunkid} = sprintf('chunk-%i.mat', chunkid);
      save(outpath, '-v7.3', 'data');
      c = 1;
      chunkid = chunkid + 1;

      master.n = n;
      save(sprintf('%s/master.mat', outdir), '-v7.3', 'master');
    end
  end
end

data = data(:, 1:c-1);
master.files{chunkid} = sprintf('chunk-%i.mat', chunkid);
fprintf('icnn: flush chunk %i\n', chunkid);
save(sprintf('%s/chunk-%i.mat', outdir, chunkid), '-v7.3', 'data');

master.n = n;
save(sprintf('%s/master.mat', outdir), '-v7.3', 'master');
