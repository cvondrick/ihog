function dblur = xpassdict(dgray, dim, k, sig),

if ~exist('sig', 'var'),
  sig = 10;
end

if sig < 0,
  sig = -sig;
  lo = true;
else,
  lo = false;
end

ny = dim(1);
nx = dim(2);
if length(dim) > 2,
  nc = dim(3);
else,
  nc = 1;
end
  
% build blurred dgray
dblur = zeros(size(dgray));
fil = fspecial('gaussian', round([ny/2 nx/2]), sig);
for i=1:k,
  elemnorm = norm(dgray(:, i));
  elem = reshape(dgray(:, i), dim); 
  for j=1:nc,
    if lo,
      elem(:, :, j) = filter2(fil, elem(:, :, j), 'same');
    else,
      elem(:, :, j) = elem(:, :, j) - filter2(fil, elem(:, :, j), 'same');
    end
  end
  dblur(:, i) = elem(:) / elemnorm;
end

if nargout == 0, 
  sy = 10;
  sx = 10;
  bord = 10;
  midpad = 1;
  cy = (ny+bord);
  cx = (nx*2+bord+midpad);
  %iii = randperm(size(dgray,2));
  im = zeros(cy*sy, cx*sx, nc);
  for i=1:min(sy*sx, k),
    row = mod(i-1, sx)+1;
    col = floor((i-1) / sx)+1;

    graypic = dgray(:, i);
    graypic = reshape(graypic, [ny nx nc]);
    graypic(:) = graypic(:) - min(graypic(:));
    graypic(:) = graypic(:) / max(graypic(:));

    blurpic = dblur(:, i);
    blurpic = reshape(blurpic, [ny nx nc]);
    blurpic(:) = blurpic(:) - min(blurpic(:));
    blurpic(:) = blurpic(:) / max(blurpic(:));

    pic = cat(2, graypic, ones(ny, midpad, nc), blurpic);
    pic = padarray(pic, [bord bord], 1, 'post');

    im((col-1)*cy+1:col*cy, (row-1)*cx+1:row*cx, :) = pic;
  end
  im = im(1:end-bord, 1:end-bord, :);
  imagesc(im);
  axis image;
end
