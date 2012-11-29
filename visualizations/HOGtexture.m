% HOGtexture(feat)
%
% Creates a visualization of the HOG texture features. For each HOG cell, it
% builds a histogram and attempts to visualize it spatially.
function bigfig = HOGtexture(feat),
texture = feat(:, :, 9*3+1:end);
[ny,nx,nf] = size(texture);

for k=1:nf,
  t = texture(:, :, k);
  texture(:, :, k) = texture(:, :, k) - min(t(:));
  t = texture(:, :, k);
  m = max(t(:));
  if m > 0,
    texture(:, :, k) = texture(:, :, k) / m;
  end
end

b = 1;
h = 8 * 3;
w = round(8 / nf) * 3;
bigfig = ones(ny*(h+b), nx*(w*nf+b), 3) * 0.75;
cc = prism(nf);

for i=1:ny,
  for j=1:nx,
    for k=1:nf,
      val = round(texture(i, j, k) * h);
      for z=1:3,
        bigfig((i-1)*(h+b)+1+(h+b-val) : i*(h+b), ...
               (j-1)*(w*nf+b)+1+(k-1)*w+1 : (j-1)*(w*nf+b)+1+k*w, z) = cc(k, z);
      end
    end
  end
end
