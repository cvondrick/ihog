function visualizeSquare(output),

dim = [80 80];
pos = [50 25];

fprintf('ihog: square: ');

c = 1;
for i=0:200;
  im = ones(200, 400, 3);

  x = floor(pos + [0 i]);

  im(x(1):x(1)+dim(1), x(2):x(2)+dim(2), :) = 0;

  vis = hoggles(im);

  imagesc(vis);
  axis image;
  drawnow;

  imwrite(vis, [output '/' sprintf('%05d.jpg', c)]);
  c = c + 1;

  fprintf('.');
end
fprintf('\n');
