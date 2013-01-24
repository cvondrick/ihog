% reportcard(in, out, pd)
%
% Processes every image in input directory 'in' and outputs
% the inversion to 'out' using the paired dictionary 'pd'. This
% is useful for diagnosis purposes.
function reportcard(in, out, pd),

errors = [];

images = dir(in);
for i=1:length(images);
  if ~images(i).isdir,
    filepath = [in '/' images(i).name];
    im = double(imread(filepath)) / 255.;
    feat = gistfeatures(im);
    [ihog, err] = invertGist(feat, pd);

    errors = [errors err];

    im = imresize(im, [size(ihog, 1) size(ihog, 2)]);
    im(im > 1) = 1;
    im(im < 0) = 0;
    im = mean(im, 3);

    graphic = cat(2, ihog, im); 

    subplot(121);
    imagesc(graphic);
    axis image;
    subplot(122);
    hist(errors);
    drawnow;

    imwrite(graphic, sprintf('%s/%s', out, images(i).name));

    fprintf('processed %s with error %f\n', filepath, err);
  end
end
