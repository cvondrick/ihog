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

hny = pd.ny;
hnx = pd.nx;
sbin = pd.sbin;

gny = (hny+2)*sbin;
gnx = (hnx+2)*sbin;

bord = 10;
cy = (gny+bord);
cx = (gnx*2+bord);

im = ones(cy*sy, cx*sx);

iii = randperm(size(pd.dgray,2));

fprintf('ihog: show pair dict: ');
for i=1:min(sy*sx, pd.k),
  fprintf('.');

  row = mod(i-1, sx)+1;
  col = floor((i-1) / sx)+1;

  graypic = reshape(pd.dgray(:, iii(i)), [gny gnx]);
  graypic(:) = graypic(:) - min(graypic(:));
  graypic(:) = graypic(:) / max(graypic(:));

  hogfeat = reshape(pd.dhog(:, iii(i)), [hny hnx 32]);
  hogpic = HOGpicture(hogfeat);
  hogpic = imresize(hogpic, [gny gnx]);
  hogpic(hogpic < 0) = 0;
  hogpic(hogpic > 1) = 1;

  pic = cat(2, graypic, hogpic);
  pic = padarray(pic, [bord bord], 1, 'post');

  im((col-1)*cy+1:col*cy, (row-1)*cx+1:row*cx) = pic;
end
fprintf('\n');

im = im(1:end-bord, 1:end-bord);

imagesc(im);
axis image;
colormap gray;
