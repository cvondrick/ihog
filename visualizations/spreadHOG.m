% spreadHOG(feat)
%
% Shows the HOG channels seperately.
function vis = spreadHOG(feat),
fprintf('ihog: spread: ');
nf = size(feat, 3);
for i=1:nf,
  fprintf('.');
  f = zeros(size(feat));
  f(:, :, i) = feat(:, :, i);
  ihog = invertHOG(f);

  if i < 28,
    f(:) = 0;
    f(:, :, mod(i-1, 9)+18) = feat(:, :, i);
    glyph = HOGpicture(f);
    glyph = imresize(glyph, size(ihog));
    gylph(glyph > 1) = 1;
    gylph(glyph < 0) = 0;
  else,
    glyph = zeros(size(ihog));
  end

  ihog = padarray(ihog, [5 5], 0.5);
  glyph = padarray(glyph, [5 5], 0.5);

  subplot(4, 9, i);
  imagesc([ihog glyph]);
  axis image;
  axis off;
end
fprintf('\n');

subplot(4,9,1); title('+ Signed');
subplot(4,9,10); title('- Signed');
subplot(4,9,19); title('Unsigned');
subplot(4,9,28); title('Texture');
