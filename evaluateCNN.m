function evaluateCNN(dirpath, outpath, pd),

warning off;
mkdir(outpath);
warning on;

files = dir(dirpath);
for i=1:length(files),
  if files(i).isdir || files(i).name(1) == '.',
    continue;
  end

  fprintf('processing %s\n', files(i).name);

  payload = load([dirpath '/' files(i).name]);
  icnn = invertCNN(permute(payload.pool5_cudanet_out, [2 3 4 1]), pd);

  height = pd.imdim(2);
  width = pd.imdim(1);

  rx = pd.imdim(2)/double(payload.cropdim(1));
  ry = pd.imdim(1)/double(payload.cropdim(2));

  reconstruction = zeros(floor(payload.imsize(2)*rx), ...
                         floor(payload.imsize(1)*ry), 3);
  weights = zeros(size(reconstruction));
  original = zeros(size(reconstruction));
  for j=1:size(icnn, 4),
    xc = payload.locations(j, 2) * rx + 1;
    yc = payload.locations(j, 1) * ry + 1;

    xxx = xc:xc+width-1;
    yyy = yc:yc+height-1;

    reconstruction(yyy, xxx, :) = reconstruction(yyy, xxx, :) + icnn(:, :, :, j);
    weights(yyy, xxx, :) = weights(yyy, xxx, :) + 1;
    
    imp = imresize(squeeze(payload.images(j, :, :, :)), pd.imdim(1:2));
    original(yyy, xxx, :) = im2double(imp);
  end

  reconstruction = reconstruction ./ weights;

  vis = cat(2, reconstruction, original);

  imwrite(vis, [outpath '/' files(i).name '.jpg']);
end
