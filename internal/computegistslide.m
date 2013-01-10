function computegistslide(source, out),

images = dir(source);
images = images(randperm(length(images)));

for i=1:length(images),
  I = images(i);

  if I.isdir,
    continue;
  end

  if exist(sprintf('%s/%s.lock', source, I.name), 'dir'),
    fprintf('igist: skipping %s because lock file exists\n', I.name);
    continue;
  end

  if exist(sprintf('%s/%s.mat', out, I.name), 'file'),
    fprintf('igist: skipping %s because output already exists', I.name);
    continue;
  end

  fprintf('igist: processing %s\n', I.name);

  im = double(imread(sprintf('%s/%s', source, I.name))) / 255.;
  im = mean(im,3);

  gdim = 128;
  skipby = 32;
  c = 1;
  for x=1:skipby:size(im,2)-gdim,
    for y=1:skipby:size(im,1)-gdim,
      crop = im(y:y+gdim-1, x:x+gdim-1);
      feat = gistfeatures(repmat(im, [1 1 3]));
      data{c} = single([crop(:); feat(:)]);
      c = c + 1;
    end
  end

  save(sprintf('%s/%s.mat', out, I.name), 'data');
end
