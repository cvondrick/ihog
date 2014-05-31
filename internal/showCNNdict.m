% showpairdict(pd, sy, sx)
%
% Visualizes a few random elements from the paired dictionaries 'pd'. The
% parameters sy and sx are optional and specify the number of elements to show.
function im = showpairdict(pd, sy, sx),

if ~exist('sy', 'var'),
  sy = 10;
end
if ~exist('sx', 'var'),
  sx = 10;
end

gny = pd.imdim(1);
gnx = pd.imdim(2);

bord = 10;
cy = (gny+bord);

if isfield(pd, 'dhog'),
  cx = (gnx*2+bord);
else,
  cx = (gnx+bord);
end

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

  if isfield(pd, 'dhog'),
    hogfeat = reshape(pd.dhog(:, iii(i)), [8 8 computeHOG()]);
    hogpic = showHOG(max(0, hogfeat));
    hogpic = imresize(hogpic, [gny gnx]);
    hogpic(hogpic < 0) = 0;
    hogpic(hogpic > 1) = 1;
    hogpic = repmat(hogpic, [1 1 3]);
    graypic = cat(2, graypic, hogpic);
  end

  pic = padarray(graypic, [bord bord], 1, 'post');

  im((col-1)*cy+1:col*cy, (row-1)*cx+1:row*cx, :) = pic;
end
fprintf('\n');

im = im(1:end-bord, 1:end-bord, :);

imagesc(im);
axis image;
colormap gray;
