% spreadHOG(feat)
%
% Shows the HOG channels seperately.
function vis = spreadHOG(feat),
bord = 10;
ny = size(feat,1)*8+16+2*bord;
nx = (size(feat,2)*8+16+bord)*2;
nf = size(feat, 3);
bigfig = ones(ny*4, nx*9) * 0.5;

fprintf('ihog: spread: ');
for i=1:nf,
  fprintf('.');
  f = zeros(size(feat));
  f(:, :, i) = feat(:, :, i);
  ihog = invertHOG(f);

  if i < 28,
    f(:) = 0;
    f(:, :, mod(i-1, 9)+18+1) = feat(:, :, i);
    glyph = HOGpicture(f);
    glyph = imresize(glyph, size(ihog));
    gylph(glyph > 1) = 1;
    gylph(glyph < 0) = 0;
  else,
    glyph = zeros(size(ihog));
  end

  im = [ihog glyph];
  im = padarray(im, [bord bord], 0.5);

  col = mod(i-1, 9);
  row = floor((i-1)/9);

  bigfig(row*ny+1:(row+1)*ny, col*nx+1:(col+1)*nx) = im;
end
fprintf('\n');

imagesc(bigfig);
axis image;
