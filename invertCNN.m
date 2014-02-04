function im = invertCNN(feat, pd),

if ~exist('pd', 'var'),
  global icnn_pd
  if isempty(icnn_pd),
    if ~exist('pd-icnn.mat', 'file'),
      error('please create pd-icnn.mat or otherwise specify pd');
    end
    icnn_pd = load('pd-icnn.mat');
  end
  pd = icnn_pd;
end

windows = zeros(prod(pd.featdim), size(feat,4));
for i=1:size(feat,4),
  elem = feat(:, :, :, i);
  elem(:) = elem(:) - mean(elem(:));
  elem(:) = elem(:) / (sqrt(sum(elem(:).^2) + eps));
  windows(:, i) = elem(:);
end

% solve lasso problem
param.lambda = pd.lambda;
param.mode = 2;
param.pos = true;
a = full(mexLasso(single(windows), pd.dcnn, param));
recon = pd.drgb * a;

%fprintf('icnn: sparsity=%0.2f\n', sum(a(:) == 0) / length(a(:)));

im = reshape(recon, [pd.imdim size(windows,2)]);
for i=1:size(feat,4),
  img = im(:, :, :, i);
  img(:) = img(:) - min(img(:));
  img(:) = img(:) / max(img(:));
  im(:, :, :, i) = img;
end
