% Visualize HOG features/weights.
%   visualizeHOG(w)
function showHOG(w)

% Make pictures of positive and negative weights
bs = 20;
pos = HOGpicture(w, bs) * 255;
neg = HOGpicture(-w, bs) * 255;

% Put pictures together and draw
buff = 10;
if min(w(:)) < 0
  pos = padarray(pos, [buff buff], 128, 'both');
  neg = padarray(neg, [buff buff], 128, 'both');
  im = uint8([pos neg]);
else
  im = uint8(pos);
end
imagesc(im); 
colormap gray;
axis image;
