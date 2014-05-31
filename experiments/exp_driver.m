function exp_driver(param, outpath, method, pd),

if ~exist('pd', 'var'),
  pd = load('pd-caffe.mat');
end
rootpath = '/data/vision/torralba/hallucination/icnn/rcnn-features/voc_2007_val';

n = 3;

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
  elseif exist(lockfile, 'dir'),
    continue;
  end
  mkdir(lockfile);

  payload = load(infile);
  im = im2double(imread(sprintf('%s/images/%s.jpg', rootpath, name)));
  gt = find(payload.gt);

  feat = cell(length(gt),1);
  orig = cell(length(gt),1);
  boxes = cell(length(gt),1);
  out = cell(length(gt),1);

  for j=1:length(gt),
    fprintf('processing %s: %i/%i\n', name, j, length(gt));

    feat{j} = payload.feat(gt(j), :)';
    boxes{j} = payload.boxes(gt(j), :);
    orig{j} = im2double(uint8(im_crop(single(im2uint8(im)), boxes{j}, 'warp', 227, 16, [])));

    out{j} = method(feat{j}, pd, n, param, [], orig{j});

    %clf;
    %imdiffmatrix(out{j}, orig{j});
    %pause;
  end

  save(outfile, 'out', 'feat', 'im', 'boxes', 'orig', 'infile', 'param');

  try,
    rmdir(lockfile);
  catch
  end
end
