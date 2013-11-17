function visualizeCNN(data, pd),

for i=1:size(data.features,1),
  subplot(121);
  imagesc(invertCNN(data.features(i, :), pd));
  axis image;

  subplot(122);
  imagesc(reshape(data.images(i, :), [256 256 3]));
  axis image;

  pause(.5);
end
