function build_pairfile,

featdim = 4096;
imagedim = 64;
imagedimorig = 256;
maxc = 100000;

features = zeros(maxc, featdim, 'single');
images   = zeros(maxc, imagedim^2 * 3, 'single');
c = 1;

files = dir('data');
iii = randperm(length(files));
for i=1:length(files),
  if files(iii(i)).isdir,
    continue;
  end

  payload = load(['data/' files(iii(i)).name]);
  num = size(payload.features,1);

  features(c:c+num-1, :) = payload.features;

  for j=1:num,
    im = reshape(payload.images(j, :), [imagedimorig imagedimorig 3]);
    im = imresize(im, [imagedim imagedim]);
    im = single(im(:)) / 255.;
    images(c+j-1, :) = im;
  end
  
  c = c + num;

  uprintf('loaded %i windows (%i of %i files)', c-1, i, length(files));

  if c > maxc,
    break;
  end
end
uprintf();

features = features(1:c-1, :);
images = images(1:c-1, :);

save('data/pairs.mat', 'features', 'images', '-v7.3');
