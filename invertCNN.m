function im = invertCNN(feat, pd),

for i=1:size(feat,2),
  feat(:, i) = feat(:, i) - mean(feat(:, i));
  feat(:, i) = feat(:, i) / (sqrt(sum(feat(:, i).^2) + eps));
end

% solve lasso problem
param.lambda = pd.lambda;
param.mode = 2;
param.pos = true;
a = full(mexLasso(single(feat), pd.dhog, param));
recon = pd.dgray * a;

im = reshape(recon, [pd.ny pd.nx 3 size(feat,2)]);
for i=1:size(feat,2),
  img = im(:, :, :, i);
  img(:) = img(:) - min(img(:));
  img(:) = img(:) / max(img(:));
  im(:, :, :, i) = img;
end
