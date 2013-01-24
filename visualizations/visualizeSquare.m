% visualizeSquare(output)
%
% This function translates a black square in a toy image, computes HOG on it,
% inverts it. Notice how this function reveals some invariances of HOG: while
% the square moves every frame, the inverse square snaps to the nearest HOG
% cell boundary.
%
% This function takes a single parameter: the directory to output frames. These
% frames can then be turned into a movie with the command:
%
%   $ ffmpeg -i %05d.jpg -b 1M out.mp4 -b 1M
%
% This function is not optimized and may be slow.

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
