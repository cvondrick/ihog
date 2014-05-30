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

  for j=1:length(refeat),
    fprintf('%s: %i/%i\n', featfiles(i).name, j, length(refeat));

    distances = zeros(size(data.out{j}, 4), 1);
    sampledistances = zeros(size(data.out{j}, 4), size(sample, 2));

    for k=1:size(data.out{j}, 4),
      distances(k) = norm(refeat{j}(:, k) - data.feat{j});

      for w=1:size(sample, 2),
        sampledistances(k, w) = norm(refeat{j}(:, k) - sample(:, w));
      end
    end

    ratios = distances / distances(1);

    sampleratios = mean(sampledistances') / distances(1);

    clf;
    subplot(121);
    imdiffmatrix(reclip(data.out{j}, 0.02), data.orig{j});

    subplot(222);
    hold on;
    plot(ratios, '.', 'MarkerSize', 50);
    plot(sampleratios, 'r^', 'MarkerSize', 20);
    plot([1 length(ratios)], [1 1], 'k-', 'LineWidth', 5);
    ylabel('Ratio Distance', 'FontSize', 20);

    subplot(224);
    hold on;
    plot(distances, 'b.', 'MarkerSize', 50);
    plot(mean(sampledistances'), 'r.', 'MarkerSize', 50);
    plot([1 length(ratios)], [distances(1) distances(1)], 'k-', 'LineWidth', 5);
    ylabel('L2 Distance', 'FontSize', 20);

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
