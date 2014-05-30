function im = reclip(im, p),

for i=1:size(im, 4),
  v = im(:, :, :, i);

  lbound = prctile(v(:), 100*p);
  ubound = prctile(v(:), 100*(1-p));

  v(:) = v(:) - lbound;
  v(:) = v(:) / (ubound - lbound);

  v(v < 0) = 0;
  v(v > 1) = 1;

  im(:, :, :, i) = v;
end
