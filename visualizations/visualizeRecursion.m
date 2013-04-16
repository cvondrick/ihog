function bigim = visualizerRecursion(im),

s = size(im);

im = im2double(im);
im = rgb2gray(im);
im = repmat(im, [1 1 3]);

feat = features(im, 8);
hogim = imresize(showHOG(feat), [s(1) s(2)]);
hogim(hogim > 1) = 1;
hogim(hogim < 0) = 0;
hogim = repmat(hogim, [1 1 3]);

bigim = padarray([im; hogim], [0 5], 1);
imagesc(bigim);
axis image;
drawnow;

for i=1:5,
  feat = features(im, 8);
  hogim = imresize(showHOG(feat), [s(1) s(2)]);
  hogim(hogim > 1) = 1;
  hogim(hogim < 0) = 0;
  hogim = repmat(hogim, [1 1 3]);

  im = repmat(invertHOG(feat), [1 1 3]);
  im = imresize(im, [s(1) s(2)]);
  im(im > 1) = 1;
  im(im < 0) = 0;

  bigim = [bigim padarray([im; hogim], [0 5], 1)];
  imagesc(bigim);
  axis image;
  drawnow;
end
