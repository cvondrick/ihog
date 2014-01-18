function convdemo(feat, ny, nx),

glyph = showHOG(feat);
glyph = imresize(glyph, [(size(feat,1)+2)*8, (size(feat,2)+2)*8]);
imwrite(glyph, '~/Desktop/slide/blank.jpg');

count = 1;

for i=1:size(feat,1)-ny,
  for j=1:size(feat,2)-nx,
    crop = feat(i:i+ny-1, j:j+nx-1, :);
    ihog = invertHOG(crop);

    vis = glyph;
    vis((i-1)*8+1:(i+ny+1)*8, (j-1)*8+1:(j+nx+1)*8) = ihog;

    imwrite(vis, sprintf('~/Desktop/slide/%i.jpg', count));
    count = count + 1;

    fprintf('.');
  end
end

