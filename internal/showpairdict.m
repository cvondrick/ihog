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

dim = pd.dim * 5;

bord = 50;
cy = (dim+bord);
cx = (dim*2+bord);

im = ones(cy*sy, cx*sx, 3);

iii = randperm(size(pd.dgray,2));
%iii = 1:size(pd.dgray,2);

fprintf('ihog: show pair dict: ');
for i=1:min(sy*sx, pd.k),
  fprintf('.');

  row = mod(i-1, sx)+1;
  col = floor((i-1) / sx)+1;

  graypic = reshape(pd.dgray(:, iii(i)), [pd.dim pd.dim]);
  graypic(:) = graypic(:) - min(graypic(:));
  graypic(:) = graypic(:) / max(graypic(:));
  graypic = imresize(graypic, [dim dim]);
  graypic(graypic > 1) = 1;
  graypic(graypic < 0) = 0;
  graypic = repmat(graypic, [1 1 3]);

  gistpic = visualizeGist(max(pd.dhog(:, iii(i))', 0));
  gistpic = imresize(gistpic, [dim dim]);
  gistpic(gistpic > 1) = 1;
  gistpic(gistpic < 0) = 0;

  pic = cat(2, graypic, gistpic);
  pic = padarray(pic, [bord bord], 1, 'post');

  im((col-1)*cy+1:col*cy, (row-1)*cx+1:row*cx, :) = pic;
end
fprintf('\n');

im = im(1:end-bord, 1:end-bord, :);

imagesc(im);
axis image;
colormap gray;
