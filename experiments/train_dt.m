cnn_cache = '/data/vision/torralba/hallucination/icnn/rcnn-features/voc_2007_train';
cache_path = '/scratch/carl/cache/icnn-dt-';
n = 10000;

try,
  load([cache_path 'cnn.mat']);
catch,
  feat = zeros(pd.featdim, n);
  classes = zeros(n,1);
  c = 1;

  cnn_files = dir(cnn_cache);
  cnn_files = cnn_files(randperm(length(cnn_files)));
  for cnn_file_id = 1:length(cnn_files),
    if cnn_files(cnn_file_id).isdir || ~strcmp(cnn_files(cnn_file_id).name(end-3:end), '.mat')
      continue;
    end

    fprintf('load %i/%i\n', c, n);

    payload = load([cnn_cache '/' cnn_files(cnn_file_id).name]);
    gtind = find(payload.gt);

    if c+length(gtind) > n,
      gtind = gtind(randperm(length(gtind), n-c+1));
    end

    feat(:, c:c+length(gtind)-1) = payload.feat(gtind, :)';

    [~, clsid] = max(payload.overlap(gtind, :)');
    classes(c:c+length(gtind)-1) = clsid;

    c = c + length(gtind);

    if c > n,
      break;
    end
  end

  keep = classes > 0;
  classes = classes(keep);
  feat = feat(:, keep);
  save([cache_path 'cnn.mat'], 'feat', 'classes');
end

try,
  load([cache_path 'icnn.mat']);
catch,
  fprintf('inverting\n');
  icnn = invertCNN(feat, pd);
  save([cache_path 'icnn.mat'], 'icnn');
end

try,
  load([cache_path 'hog.mat']);
catch,
  sbin = 8;
  hog = zeros(pd.imdim(1) / sbin - 2, pd.imdim(2) / sbin - 2, computeHOG(), size(icnn,4));
  for i=1:size(icnn, 4),
    fprintf('compute HOG %i/%i\n', i, size(icnn, 4));
    hog(:, :, :, i) = computeHOG(double(icnn(:, :, :, i)), sbin);
  end
  save([cache_path 'hog.mat'], 'hog');
end

try,
  fail
  load([cache_path 'model.mat']),
catch,
  uclasses = unique(classes);
  w = zeros(size(hog,1)*size(hog,2)*size(hog,3), length(uclasses));
  b = zeros(length(uclasses), 1);
  for cls=uclasses',
    fprintf('training class %i\n', cls);

    clsind = (classes == cls);
    X = cat(2, reshape(hog(:, :, :, clsind), [], sum(clsind)), ...
              reshape(hog(:, :, :, ~clsind), [], sum(~clsind)));
    X = X';
    Y = [ones(sum(clsind), 1); -ones(sum(~clsind),1)];

    model = svmtrain(Y, X, '-s 0 -t 0 -c .01');
    [~, accuracy, ~] = svmpredict(Y, X, model);
    fprintf('training accuracy is %f\n', accuracy(1));

    w(:, cls) = model.SVs' * model.sv_coef;
    b(cls) = -model.rho;

    showHOG(reshape(w(:, cls), [size(hog,1) size(hog,2) size(hog,3)]));
    drawnow;
  end

  save([cache_path 'model.mat'], 'w', 'b'),
end
