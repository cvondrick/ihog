function [inverse, original] = visualizeCNN(data, pd, nx, ny),

skip = 4;

inverse = zeros(64+(ny-1)*skip, 64+(nx-1)*skip, 3);
original = zeros(64+(ny-1)*skip, 64+(nx-1)*skip, 3);
weights = zeros(64+(ny-1)*skip, 64+(nx-1)*skip);

icnns = invertCNN(data.features', pd);

c = 1;
for i=1:ny,
  for j=1:nx,
    icnn = icnns(:, :, :, c);
    orig = reshape(data.images(c, :), [256 256 3]);
    orig = im2double(imresize(orig, [64 64]));
    iii = (i-1)*skip+1:(i-1)*skip+64;
    jjj = (j-1)*skip+1:(j-1)*skip+64;
    inverse(iii, jjj, :) = inverse(iii, jjj, :) + icnn;
    original(iii, jjj, :) = original(iii, jjj, :) + orig;
    weights(iii, jjj) = weights(iii, jjj) + 1;
    c = c + 1;

%    subplot(131);
%    imagesc(icnn);
%    axis image;
%    subplot(132);
%    imagesc(orig);
%    axis image;
%    title(sprintf('%i', c-1));
%    subplot(133);
%    imagesc(original ./ repmat(weights, [1 1 3]));
%    pause;
 end
end

weights = repmat(weights, [1 1 3]);
inverse = inverse ./ weights;
original = original ./ weights;

if nargout == 0,
  clf;
  subplot(121);
  imagesc(inverse);
  axis image;
  subplot(122);
  imagesc(original);
  axis image;
end
