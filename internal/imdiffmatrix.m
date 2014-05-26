function im = imdiffmatrix(ims, orig, bord, render),

if ~exist('bord', 'var'),
  bord = 5;
end
if ~exist('render', 'var'),
  render = @(x) x;
end

[h, w, nc] = size(render(ims(:, :, :, 1)));
n = size(ims, 4);

im = ones(h*(n+1), w*(n+1), nc);

if exist('orig', 'var') && ~isempty(orig),
  orig = imresize(render(orig), [h w]);
  orig(orig > 1) = 1;
  orig(orig < 0) = 0;
  if nc == 1,
    orig = mean(orig, 3);
  end
  orig = padarray(orig, [bord bord], .8);
else,
  orig = 0.8 * ones(h+2*bord, w+2*bord, 3);
end

h = h + 2 * bord;
w = w + 2 * bord;

% build borders
for i=1:n,
  im(h*i:h*(i+1)-1, 1:w, :) = padarray(render(ims(:, :, :, i)), [bord bord 0], .8);
  im(1:h, w*i:w*(i+1)-1, :) = padarray(render(ims(:, :, :, i)), [bord bord 0], .8);
end

im(1:h, 1:w, :) = orig;

for i=1:n,
  for j=1:n,
    d = abs(ims(:, :, :, i) - ims(:, :, :, j));
    d(:) = d(:) * 2;
    d = min(d, 1);
    d = render(d);
    d = mean(d, 3);
    d = padarray(d, [bord bord 0], 1);
    im(h*j:h*(j+1)-1, w*i:w*(i+1)-1, :) = repmat(d, [1 1 nc]);
  end
end

if nc==1,
  im = repmat(im, [1 1 3]);
end

if nargout == 0,
  imagesc(im);
  axis image;
  axis off;
end
