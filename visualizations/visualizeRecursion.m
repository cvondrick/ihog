function visualizerRecursion(im),

im = im2double(im);

origfeat = features(im, 8);

history = [];

subplot(321);
feat = features(im, 8);
showHOG(feat);

subplot(322);
imagesc(invertHOG(feat));
axis image;

for i=1:100,
  feat = features(im, 8);
  im = repmat(invertHOG(feat), [1 1 3]);

  subplot(323);
  showHOG(feat);

  subplot(324);
  imagesc(im); axis image;

  subplot(325);
  showHOG(feat - origfeat);

  subplot(326);
  history = [history norm(origfeat(:) - feat(:))];
  plot(history);

  drawnow;
end
