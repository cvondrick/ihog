function show_results(datapath, pd),

sample = load_sample(1000);

featfiles = dir([datapath '/feat']);
for i=1:length(featfiles),
  if featfiles(i).isdir,
    continue;
  end

  feat = load([datapath '/feat/' featfiles(i).name]);
  refeat = feat.feat;
  data = load([datapath '/' featfiles(i).name]);

  invsample = zeros(9216, 10);
  c = 1;
  jjj = randperm(length(featfiles));
  while true,
    j = ceil(rand() * length(featfiles));
    if featfiles(j).isdir || j==i,
      continue;
    end
    samplefeat = load([datapath '/feat/' featfiles(j).name]);
    whichindex = ceil(rand() * length(samplefeat.feat));
    whichindex2 = ceil(rand() * size(samplefeat.feat{whichindex}, 2));
    invsample(:, c) = samplefeat.feat{whichindex}(:, whichindex2);
    c = c + 1;
    if c > size(invsample,2),
      break;
    end
  end

  for j=1:length(refeat),
    fprintf('%s: %i/%i\n', featfiles(i).name, j, length(refeat));

    featinv = refeat{j};

    distances = zeros(size(data.out{j}, 4), 1);
    sampledistances = zeros(size(data.out{j}, 4), size(sample, 2));
    invsampledistances = zeros(size(data.out{j}, 4), size(invsample, 2));

    for k=1:size(data.out{j}, 4),
      distances(k) = norm(featinv(:, k) - data.feat{j});

      for w=1:size(sample, 2),
        sampledistances(k, w) = norm(data.feat{j} - sample(:, w));
      end
      for w=1:size(invsample, 2),
        invsampledistances(k, w) = norm(data.feat{j} - invsample(:, w));
      end
    end

    ratios = distances / distances(1);

    sampleratios = mean(sampledistances') / distances(1);
    invsampleratios = mean(invsampledistances') / distances(1);

    clf;
    subplot(121);
    imdiffmatrix(data.out{j}, data.orig{j});

    subplot(4,2,2);
    hold on;
    plot(ratios, '.', 'MarkerSize', 50);
    plot(sampleratios, 'r.', 'MarkerSize', 50);
    plot(invsampleratios, 'k.', 'MarkerSize', 50);
    plot([1 length(ratios)], [1 1], 'k-', 'LineWidth', 5);
    ylabel('Ratio Distance', 'FontSize', 20);
    ylim([0.7 1.3]);

    subplot(4,2,4);
    hold on;
    plot(distances, 'b.', 'MarkerSize', 50);
    plot(mean(sampledistances'), 'r.', 'MarkerSize', 50);
    plot(mean(invsampledistances'), 'k.', 'MarkerSize', 50);
    plot([1 length(ratios)], [distances(1) distances(1)], 'k-', 'LineWidth', 5);
    ylabel('L2 Distance', 'FontSize', 20);

    subplot(247);
    imagesc(squareform(pdist(reshape(data.out{j}, [], size(data.out{j},4))')));
    axis image;
    colorbar();
    title('Pairwise Distance in Image Space', 'FontSize', 20);
    subplot(248);
    imagesc(squareform(pdist(refeat{j}')))
    axis image;
    colorbar();
    title('Pairwise Distance in Feature Space', 'FontSize', 20);

    pause;
  end
end



function sample = load_sample(n),

cachefile = '/scratch/carl/icnn-sample.mat';
try,
  load(cachefile);
catch,
  filepath = '/data/vision/torralba/hallucination/icnn/rcnn-features/voc_2007_train';
  files = dir(filepath);
  files = files(randperm(length(files)));
  sample = zeros(9216, n);
  n_store = n;

  fprintf('loading sample: ');
  c = 1;
  while true,
    i = ceil(rand() * length(files));
    if files(i).isdir,
      continue;
    end
    if ~strcmp(files(i).name(end-3:end), '.mat'),
      continue;
    end

    fprintf('.');

    p = load([filepath '/' files(i).name]);

    sample(:, c) = p.feat(ceil(rand() * size(p.feat,1)), :);
    c = c + 1;
    if c > n,
      break;
    end
  end
  fprintf('\n');

  save(cachefile, 'sample');
end
