pd = load('pd-caffe.mat');
rootpath = '/data/vision/torralba/hallucination/icnn/rcnn-features/voc_2007_val';

param.mode = 'rgb';
param.gam = 10;
param.slices = 1;
n = 10;

outpath = sprintf('/data/vision/torralba/hallucination/icnn/experiments/%s-%i', param.mode, param.gam);


files = dir(rootpath);

for iter=1:1000,
  i = floor(rand() * length(files) + 1);

  if files(i).isdir || ~strcmp(files(i).name(end-3:end), '.mat'),
    continue;
  end

  [~, name] = fileparts(files(i).name);

  infile = [rootpath '/' name '.mat'];
  outfile = [outpath '/' name '.mat'];
  lockfile = [outfile '.lock'];

  if exist(outfile, 'file'),
    continue;
  elseif exist(lockfile, 'file'),
    continue;
  end
  mkdir(lockfile);

  payload = load(infile);
  im = im2double(imread(sprintf('%s/images/%s.jpg', rootpath, name)));
  gt = find(payload.gt);

  feat = cell(length(gt),1);
  orig = cell(length(gt),1);
  out = cell(length(gt),1);

  for j=1:length(gt),
    feat{j} = payload.feat(gt(j), :)';
    bbox = payload.boxes(gt(j), :);
    orig{j} = im(bbox(2):bbox(4), bbox(1):bbox(3), :);

    out{j} = equivCNN(feat{j}, pd, n, param, [], orig{j});
  end

  save(outfile, 'out', 'feat', 'orig', 'infile', 'param');

  try,
    rmdir(lockfile);
  catch
  end
end
