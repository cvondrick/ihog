function evaluateCNN(featpath, outpath, pd),

warning off;
mkdir(outpath);
warning on;

maxper = 100;

files = dir(featpath);
for i=1:length(files),
  if files(i).isdir || files(i).name(1) == '.' || ~strcmp(files(i).name(end-3:end), '.mat'),
    continue;
  end

  fprintf('processing %s\n', files(i).name);

  payload = load([featpath '/' files(i).name]);

  n = size(payload.feat, 1);
  if n > maxper,
    iii = randperm(n, maxper);
  else,
    iii = randperm(n);
  end

  feat = payload.feat(iii, :);
  boxes = payload.boxes(iii, :);

  icnn = invertCNN(feat', pd);

  [~, basename] = fileparts(files(i).name);
  im = im2double(imread([featpath '/images/' basename '.jpg']));

  height = pd.imdim(2);
  width = pd.imdim(1);

  reconstructions = cell(size(icnn, 4), 1);
  for i=1:size(icnn, 4),
    vis = icnn(:, :, :, i);

    bbox = boxes(i, :);
    orig = im(bbox(2):bbox(4), bbox(1):bbox(3), :);
    orig = imresize(orig, pd.imdim(1:2));
    orig(orig > 1) = 1;
    orig(orig < 0) = 0;

    reconstructions{i} = cat(2, vis, orig);
  end

  graphic = montage(reconstructions, 10, 10);
  imwrite(graphic, [outpath '/' basename '.jpg']);
  imagesc(graphic);
  drawnow;
end



% M = montage(images[, cy, cx[, pad]])
%
% Takes the cell array 'images' and displays the first
% cy*cx images of them on a grid, with 'pad' pixels in between.
function M = montage(images, cy, cx, pad),

if isnumeric(images),
  imagescell = cell(size(images, 4), 1);
  for i=1:size(images,3),
    imagescell{i} = images(:, :, i);
  end
  images = imagescell;
end

if ~exist('cy', 'var'),
  cy = floor(sqrt(length(images)));
end
if ~exist('cx', 'var'),
  cx = ceil(sqrt(length(images)));
end
if ~exist('pad', 'var'),
  pad = 5;
end

if cy < 0 && cx > 0,
  cy = ceil(length(images) / cx);
end
if cx < 0 && cy > 0,
  cx = ceil(length(images) / cy);
end

ny = size(images{1}, 1)+pad*2;
nx = size(images{1}, 2)+pad*2;
nf = size(images{1}, 3);

M = ones(ny*cy, nx*cx, nf);
c = 1;
for j=1:cy,
  for i=1:cx,
    im = padarray(images{c}, [pad pad 0], 1);
    im = imresize(im, [ny nx]);
    M((j-1)*ny+1:j*ny, (i-1)*nx+1:i*nx, :) = im;
    c = c + 1;
    if c > length(images),
      return;
    end
  end
end
