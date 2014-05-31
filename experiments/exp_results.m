function exp_results(dirpath),

samplesize = 100;

plotnames = cell(10000000,1);
featdist = zeros(length(plotnames),1);
imdist = zeros(length(plotnames),1);
c = 1;

dirs = dir(dirpath);
dirs = dirs(randperm(length(dirs)));
for d=1:length(dirs),
  if ~dirs(d).isdir || dirs(d).name(1) == '.',
    continue;
  end

  if length(strfind(dirs(d).name, 'reclip')),
    fprintf('skip %s\n', dirs(d).name);
    continue;
  end

  pos = strfind(dirs(d).name, '_');
  base = dirs(d).name(1:pos-1);

  files = dir([dirpath '/' dirs(d).name '/feat']);
  if length(files) > samplesize,
    files = files(randperm(length(files), samplesize));
  end

  data = cell(length(files), 1);
  parfor f=1:length(data),
    if files(f).isdir,
      continue;
    end
    fprintf('loading %s: %s %i/%i\n', dirs(d).name, files(f).name, f, length(files));
    feat = load([dirpath '/' dirs(d).name '/feat/' files(f).name]);
    payload = load([dirpath '/' dirs(d).name '/' files(f).name]);
    payload.refeat = feat.feat;
    data{f} = payload;;
  end

  for f=1:length(files),
    if files(f).isdir,
      continue;
    end

    fprintf('processing %s: %s %i/%i\n', dirs(d).name, files(f).name, f, length(files));
    
    for k=1:length(data{f}.refeat),
      plotnames{c} = base;
      featdist(c) = norm(data{f}.refeat{k}(:, 1) - data{f}.refeat{k}(:, 2));
      imdiff = data{f}.out{k}(:, :, :, 1) - data{f}.out{k}(:, :, :, 2);
      imdist(c) = norm(imdiff(:));
      c = c + 1;
    end
  end

  plotdata(plotnames(1:c-1), featdist(1:c-1), imdist(1:c-1));
end



function plotdata(plotnames, featdist, imdist),

clf;
hold on;

uplotnames = unique(plotnames);
colors = hsv(length(uplotnames)+1);

legends = zeros(length(uplotnames), 1);
legendnames = cell(length(uplotnames), 1);

for i=1:length(uplotnames),
  active = strcmp(plotnames, uplotnames{i});

  feat = featdist(active);
  im = imdist(active);

  [~, iii] = sort(feat);

  k = 10;
  smoothfeat = min(feat):k:max(feat);
  smoothim = zeros(size(smoothfeat));
  for j=1:length(smoothfeat),
    jjj = (feat>=smoothfeat(j)).*(feat<smoothfeat(j)+k);
    smoothim(j) = median(im(logical(jjj)));
  end
  
  legends(i) = plot(smoothfeat, smoothim, '.-', 'color', colors(i, :), 'MarkerSize', 20, 'LineWidth', 5);
  plot(feat, im, '.', 'color', colors(i, :), 'MarkerSize', 7);

  bestfit = polyfit(feat, im, 1);
  legendnames{i} = sprintf('%s = %f', uplotnames{i}, bestfit(1));

%  bestfit = polyfit(feat(iii), im(iii), 2);
%  bestfitres = linspace(min(feat), max(feat), 10);
%  plot(bestfitres, polyval(bestfit, bestfitres), '-', 'color', colors(i, :), 'LineWidth', 5);
end

legend(legends, legendnames, 'FontSize', 20);
xlabel('Feat Distance', 'FontSize', 20);
ylabel('Image Distance', 'FontSize', 20);

xlim([0 max(featdist)]);
ylim([0 max(imdist)]);

drawnow;
