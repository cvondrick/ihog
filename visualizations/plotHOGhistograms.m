function bigim = plotHOGhistograms(feat),

feat(:) = feat(:) - min(feat(:));
feat(:) = feat(:) / (1.1*max(feat(:)));

[ny,nx,nf] = size(feat);
nf = nf - 1;

height = 2*50;
width = 2*floor(50/nf);
scheme = bone(nf+1);
kkk = randperm(nf);
kkk = 1:nf;

bigim = [];

for i=1:ny,
  rowim = [];
  for j=1:nx,
    cellim = [];
    for k=1:nf,
      dimim = 0.5*ones(height, width, 3);
      dimim(:, :, 1) = scheme(kkk(k), 1);
      dimim(:, :, 2) = scheme(kkk(k), 2);
      dimim(:, :, 3) = scheme(kkk(k), 3);
      v = feat(i, j, k);
      s = .35;
      v = (v + s) / (1 + s);
      v = floor((1 - v) * height);
      dimim(1:v, :, :) = 1;
      cellim = [cellim dimim];
    end
    cellim = padarray(cellim, [2 2], 0);
    rowim = [rowim cellim];
  end
  bigim = [bigim; rowim];
end
bigim = padarray(bigim, [2 2], 0);

imagesc(bigim);
axis image;
