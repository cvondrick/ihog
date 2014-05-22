function buildCNNchunks(sourcedir, outdir, chunksize, hogdim),

if ~exist('chunksize', 'var'),
  chunksize = 100000;
end
if ~exist('featdim', 'var'),
  featdim = 9216;
end
if ~exist('imagedim', 'var'),
  imagedim = 64;
end
if ~exist('hogdim', 'var'),
  hogdim = [0 0];
end

data = zeros(prod(featdim)+imagedim^2*3+prod(hogdim)*computeHOG(), chunksize, 'single');
c = 1;
chunkid = 1;
n = 0;

master.imdim = [imagedim imagedim 3];
master.featdim = featdim;
master.hogdim = hogdim;

warning off; mkdir(outdir); warning on;

files = dir(sourcedir);
files = files(randperm(length(files)));
for i=1:length(files),
  if files(i).isdir,
    continue;
  end
  if ~strcmp(files(i).name(end-3:end), '.mat'),
    continue;
  end

  fprintf('icnn: load: %s (chunk has %i/%i or %.1f%%)\n', files(i).name, c, chunksize, c / chunksize * 100);

  [~, basename] = fileparts(files(i).name);
  imfull = imread([sourcedir '/images/' basename '.jpg']);

  payload = load([sourcedir '/' files(i).name]);

  num = size(payload.feat,1);
  n = n + num;

  for j=1:num,
    xmin = payload.boxes(j, 1);
    ymin = payload.boxes(j, 2);
    xmax = payload.boxes(j, 3);
    ymax = payload.boxes(j, 4);

    im = imfull(ymin:ymax, xmin:xmax, :);
    im = imresize(im, [imagedim imagedim]);
    im = single(im) / 255.;

    if prod(hogdim) > 0,
      hogpatch = double(imresize(im, (hogdim+2)*8));
      hog = computeHOG(hogpatch, 8);
      hog(:) = hog(:) - mean(hog(:));
      hog(:) = hog(:) / (sqrt(sum(hog(:).^2) + eps));
      data(imagedim^2*3+prod(featdim)+1:end, c) = hog(:);
    end

    feat = payload.feat(j, :);

    % normalize
    im(:) = im(:) - mean(im(:));
    im(:) = im(:) / (sqrt(sum(im(:).^2) + eps));

    feat(:) = feat(:) - mean(feat(:));
    feat(:) = feat(:) / (sqrt(sum(feat(:).^2) + eps));

    data(1:imagedim^2*3, c) = im(:);
    data(imagedim^2*3+1:imagedim^2*3+prod(featdim), c) = feat(:);

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
