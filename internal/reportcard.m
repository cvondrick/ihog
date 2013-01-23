% reportcard(in, out, pd)
%
% Processes every image in input directory 'in' and outputs
% the inversion to 'out' using the paired dictionary 'pd'. This
% is useful for diagnosis purposes.
function reportcard(in, out, pd),

images = dir([in '/*.jpg']);
images = images(randperm(length(images)));
for i=1:length(images);
  if ~images(i).isdir,

    if exist([out '/' images(i).name], 'file'),
      fprintf('ihog: %s already finished\n', images(i).name);
      continue;
    end

    if exist([in '/' images(i).name '.lock']),
      fprintf('ihog: %s is locked\n', images(i).name);
      continue;
    end
    mkdir([in '/' images(i).name '.lock']);

    filepath = [in '/' images(i).name];
    im = double(imread(filepath)) / 255.;
    im(im > 1) = 1;
    im(im < 0) = 0;
    feat = features(im, 8);
    ihog = invertHOG(feat, pd);

    if length(size(ihog)) ~= 3,
      ihog = repmat(ihog, [1 1 3]);
    end

    im = imresize(im, [size(ihog, 1) size(ihog, 2)]);
    im(im > 1) = 1;
    im(im < 0) = 0;

    graphic = cat(2, ihog, im); 

    imagesc(graphic);
    axis image;
    drawnow;

    imwrite(graphic, sprintf('%s/%s', out, images(i).name));

    fprintf('ihog: processed %s\n', filepath);

    try,
      rmdir([in '/' images(i).name '.lock']);
    catch
    end
  end
end
