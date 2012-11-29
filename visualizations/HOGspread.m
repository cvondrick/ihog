% HOGspread(feat)
function vis = HOGspread(feat, o),
vis = zeros(0, 0);
for i=o+1:o+9,
  f = zeros(size(feat));
  f(:, :, i-o) = feat(:, :, i);
  f = HOGpicture(f);
  f = padarray(f, [5 5], 0.5, 'both');
  vis = [vis f];
end
