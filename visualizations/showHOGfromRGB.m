% showHOGfromRGB()
%
% Prmopts the user to enter a color, then attempts to reconstruct that color
% patch to show what HOG is likely correlated with that color.
function showHOGfromRGB(pd, color)

if ~exist('color', 'var'),
  color = uisetcolor();
end

im = zeros((pd.ny+2)*pd.sbin, (pd.nx+2)*pd.sbin, 3);
im(:, :, 1) = color(1);
im(:, :, 2) = color(2);
im(:, :, 3) = color(3);

clf;
subplot(121);
imagesc(im);
axis image;
drawnow;

% solve lasso problem
fprintf('ihog: lasso\n');
param.lambda = pd.lambda;
param.mode = 2;
a = full(mexLasso(single(im(:)), pd.dgray, param));
recon = pd.dhog * a;

recon = reshape(recon, [pd.ny pd.nx features]);
glyph = showHOG(recon);

subplot(122);
imagesc(glyph);
axis image;
