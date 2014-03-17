function im = imdiffmatrix(ims, orig, bord),

if ~exist('bord', 'var'),
  bord = 5;
end

[h, w, n] = size(ims);
im = ones(h*(n+1), w*(n+1));

if exist('orig', 'var') && ~isempty(orig),
  orig = imresize(orig, [h w]);
  orig(orig > 1) = 1;
  orig(orig < 0) = 0;
  orig = mean(orig, 3);
  orig = padarray(orig, [bord bord], .5);
else,
  orig = 0.5 * ones(h+2*bord, w+2*bord);
end

h = h + 2 * bord;
w = w + 2 * bord;

% build borders
for i=1:n,
  im(h*i:h*(i+1)-1, 1:w) = padarray(ims(:, :, i), [bord bord], .5);
  im(1:h, w*i:w*(i+1)-1) = padarray(ims(:, :, i), [bord bord], .5);
end

im(1:h, 1:w) = orig;

for i=1:n,
  for j=1:n,
    d = abs(ims(:, :, i) - ims(:, :, j));
    d(:) = d(:) * 2;
    d = min(d, 1);
    d = padarray(d, [bord bord], 1);
    im(h*j:h*(j+1)-1, w*i:w*(i+1)-1) = d;
  end
end

im = repmat(im, [1 1 3]);

if nargout == 0,
  imagesc(im);
  axis image;
end
