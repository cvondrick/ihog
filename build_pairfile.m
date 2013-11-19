function build_pairfile,

rootdir = '/data/vision/torralba/hallucination/data-pool5';
outdir = '/data/vision/torralba/hallucination/data-pool5';
featdim = 9216;
imagedim = 64;
imagedimorig = 256;
maxc = 100000;

imsize = imagedim^2*3;

space = (imsize+featdim)*maxc*4 / 1024 / 1024 / 1024;
fprintf('allocating %0.2f GB in 3 seconds... ', space);
pause(3);
fprintf('go!!!!\n');

data = zeros(maxc, imsize + featdim, 'single');
c = 1;

files = dir(rootdir);
iii = randperm(length(files));
for i=1:length(files),
  if files(iii(i)).isdir,
    continue;
  end

  payload = load([rootdir '/' files(iii(i)).name]);
  num = size(payload.features,1);

  take = num;
  if c + num > maxc,
    take = maxc - c + 1; 
    fprintf('truncating from %i to %i\n', num, take);
  end

  try,
    for j=1:take,
      im = reshape(payload.images(j, :), [imagedimorig imagedimorig 3]);
      im = imresize(im, [imagedim imagedim]);
      im = single(im(:)) / 255.;
      data(c+j-1, 1:imsize) = im;
    end
  catch,
    fprintf('skipping %s\n', files(iii(i)).name);
    continue
  end

  data(c:c+take-1, imsize+1:end) = payload.features(1:take, :);
  
  c = c + take;

  fprintf('loaded %i windows (%i of %i files)\n', c-1, i, length(files));

  if c > maxc,
    break;
  end
end

data = data(1:c-1, :);

fprintf('final: %i windows\n', size(data,1));

save([outdir '/pairs.mat'], 'data', '-v7.3');
