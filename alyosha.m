function im = alyosha(filepath, out, pd),

rng('shuffle');

images = dir(filepath);
images = images(randperm(length(images)));

for i=1:length(images),
  if images(i).isdir,
    continue;
  end

  fprintf('%s: ', images(i).name);

  if exist([filepath '/' images(i).name '.lock'], 'dir'),
    fprintf('locked\n');
    continue;
  end
  if exist([out '/' images(i).name], 'file'),
    fprintf('completed\n');
    continue;
  end
  mkdir([filepath '/' images(i).name '.lock']);
  fprintf('working...');

  im = imread([filepath '/' images(i).name]);
  im = im2double(im);

  feat = features(im, 8);
  ihog = invertHOG(feat, pd);

  im = imresize(im, [size(ihog, 1) size(ihog, 2)]);
  im(im > 1) = 1;
  im(im < 0) = 0;

  origalt = rgb2hsv(im);

  alt = rgb2hsv(ihog);
  alt(:, :, 3) = origalt(:, :, 3);

  im = hsv2rgb(alt);

  imagesc(im); axis image; drawnow;

  imwrite(im, [out '/' images(i).name]);
  fprintf('processed!\n');

  try,
    rmdir([filepath '/' images(i).name '.lock']);
  catch
  end
end
