% showpairdict(pd, sy, sx)
%
% Visualizes a few random elements from the paired dictionaries 'pd'. The
% parameters sy and sx are optional and specify the number of elements to show.
function im = showpairdict(pd, sy, sx),

if ~exist('sy', 'var'),
  sy = 10;
end
if ~exist('sx', 'var'),
  sx = 20;
end

gny = pd.ny; 
gnx = pd.nx;

bord = 10;
cy = (gny+bord);
cx = (gnx+bord);

im = ones(cy*sy, cx*sx, 3);

iii = randperm(size(pd.drgb,2));

fprintf('icnn: show pair dict: ');
for i=1:min(sy*sx, pd.k),
  fprintf('.');

  row = mod(i-1, sx)+1;
  col = floor((i-1) / sx)+1;

  graypic = reshape(pd.drgb(:, iii(i)), [gny gnx 3]);
  graypic(:) = graypic(:) - min(graypic(:));
  graypic(:) = graypic(:) / max(graypic(:));

  pic = padarray(graypic, [bord bord], 1, 'post');

  im((col-1)*cy+1:col*cy, (row-1)*cx+1:row*cx, :) = pic;
end
fprintf('\n');

im = im(1:end-bord, 1:end-bord, :);

imagesc(im);
axis image;
colormap gray;
