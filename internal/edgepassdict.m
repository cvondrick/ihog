function dblur = xpassdict(dgray, dim),

k = size(dgray, 2);

ny = dim(1);
nx = dim(2);
if length(dim) > 2,
  nc = dim(3);
else,
  nc = 1;
end

    
% build edge
dblur = zeros(dim(1)*dim(2)*2, size(dgray,2));
for i=1:k,
  elem = reshape(dgray(:, i), dim); 
  dy = zeros(dim);
  dx = zeros(dim);
  for j=1:nc,
    dy(:, :, j) = filter2([-1 0 1], elem(:, :, j), 'same');
    dx(:, :, j) = filter2([-1 0 1]', elem(:, :, j), 'same');
  end
  dy = max(dy, [], 3);
  dx = max(dx, [], 3);
  dblur(:, i) = [dy(:); dx(:)];
end

if nargout == 0, 
  sy = 10;
  sx = 5;
  bord = 10;
  midpad = 1;
  cy = (ny+bord);
  cx = (nx*3+bord+midpad*2);
  im = zeros(cy*sy, cx*sx, nc);
  for i=1:min(sy*sx, k),
    row = mod(i-1, sx)+1;
    col = floor((i-1) / sx)+1;

    graypic = dgray(:, i);
    graypic = reshape(graypic, [ny nx nc]);
    graypic(:) = graypic(:) - min(graypic(:));
    graypic(:) = graypic(:) / max(graypic(:));

    dypic = dblur(1:dim(1)*dim(2), i);
    dypic = reshape(dypic, [ny nx 1]);
    dypic(:) = dypic(:) - min(dypic(:));
    dypic(:) = dypic(:) / max(dypic(:));
    dypic = repmat(dypic, [1 1 nc]);

    dxpic = dblur(dim(1)*dim(2)+1:end, i);
    dxpic = reshape(dxpic, [ny nx 1]);
    dxpic(:) = dxpic(:) - min(dxpic(:));
    dxpic(:) = dxpic(:) / max(dxpic(:));
    dxpic = repmat(dxpic, [1 1 nc]);

    pic = cat(2, graypic, ones(ny, midpad, nc), dypic, ones(ny, midpad, nc), dxpic);
    pic = padarray(pic, [bord bord], 1, 'post');

    im((col-1)*cy+1:col*cy, (row-1)*cx+1:row*cx, :) = pic;
  end
  im = im(1:end-bord, 1:end-bord, :);
  imagesc(im);
  axis image;
end
