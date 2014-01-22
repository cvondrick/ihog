% visualizeHOGbatch(inframes, outframes)
%
% This function is designed to run on a cluster. It processes all the images in
% the input directory, and outputs visualizations in the out frames directory.
% It processes frames in random orders and creates lock files so you can invoke
% this process many times without duplicating work.
function visualizeHOGbatch(inframes, outframes),

files = dir(inframes);
iii = randperm(length(files));
for i=1:length(files);
  if files(iii(i)).isdir,
    continue;
  end

  name = files(iii(i)).name;
  filepath = [inframes '/' name];
  output = [outframes '/' name];

  if exist([outframes '/im-' name], 'file'),
    fprintf('ihog: %s already finished\n', name);
    continue;
  end

  if exist([filepath '.lock']),
    fprintf('ihog: %s is locked\n', name);
    continue;
  end
  mkdir([filepath '.lock']);
  
  fprintf('ihog: process %s\n', name);

  im = imread(filepath);
  im = double(im) / 255.;
  feat = features(im, 8);
  ihog = invertHOG(feat);
  glyph = HOGpicture(feat);

  ihog = imresize(ihog, [size(im,1) size(im,2)]);
  glyph = imresize(glyph, [size(im,1) size(im,2)]);

  ihog(ihog > 1) = 1;
  ihog(ihog < 0) = 0;
  glyph(glyph > 1) = 1;
  glyph(glyph < 0) = 0;

  ihog = repmat(ihog, [1 1 3]);
  glyph = repmat(glyph, [1 1 3]);

  imwrite(im, [outframes '/im-' name]);
  imwrite(ihog, [outframes '/ihog-' name]);
  imwrite(glyph, [outframes '/glyph-' name]);

  try,
    rmdir([filepath '.lock']);
  catch
  end
end
