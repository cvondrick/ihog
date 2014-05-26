function [ims, orig] = exp_equiv(name, i, pd, n, param, w),

rootpath = '/data/vision/torralba/hallucination/icnn/rcnn-features/voc_2007_val';

if ~isstr(name),
  files = dir(rootpath);
  count = 0;
  for j=1:length(files),
    if ~files(j).isdir && strcmp(files(j).name(end-3:end), '.mat'),
      count = count + 1;
      if count == name,
        name = files(j).name(1:end-4);
        break;
      end
    end
  end
end

payload = load(sprintf('%s/%s.mat', rootpath, name));
im = im2double(imread(sprintf('%s/images/%s.jpg', rootpath, name)));

if i < 0,
  iii = find(payload.gt);
  i = iii(-i);
end

feat = payload.feat(i, :);
bbox = payload.boxes(i, :);
orig = im(bbox(2):bbox(4), bbox(1):bbox(3), :);

subplot(121);
[ims, vis] = equivCNN(feat', pd, n, param, w, orig);

subplot(122);
hogs = zeros(8, 8, 32, size(ims, 4));
for i=1:size(ims, 4),
  hogs(:, :, :, i) = computeHOG(imresize(ims(:, :, :, i), [80 80]), 8);
end
render = @(x) showHOG(max(0, x - mean(x(:))));
%render = @(x) invertHOG(x); 
imdiffmatrix(hogs, computeHOG(imresize(orig, [80 80]), 8), 5, render);
