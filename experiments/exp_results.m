function exp_results(dirpath),

samplesize = 100;
dohog = false;

plotnames = cell(10000000,1);
featdist = zeros(length(plotnames),1);
featratio = zeros(length(plotnames),1);
imdist = zeros(length(plotnames),1);
c = 1;

ctransform = makecform('srgb2lab');

dirs = dir(dirpath);
dirs = dirs(randperm(length(dirs)));
for d=1:length(dirs),
  if ~dirs(d).isdir || dirs(d).name(1) == '.',
    continue;
  end

  if length(strfind(dirs(d).name, 'reclip=0.02')),
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

      featdist(c) = norm(data{f}.refeat{k}(:, 1) - data{f}.refeat{k}(:, 2)) / size(data{f}.feat{1},1);
      featratio(c) = norm(data{f}.refeat{k}(:, 2) - data{f}.feat{k}) / (eps + norm(data{f}.refeat{k}(:, 1) - data{f}.feat{k}));

      if dohog,
        imdiff = computeHOG(double(data{f}.out{k}(:, :, :, 1)), 4) - computeHOG(double(data{f}.out{k}(:, :, :, 2)), 4);
      else,
        imdiff = applycform(double(data{f}.out{k}(:, :, :, 1)), ctransform) - applycform(double(data{f}.out{k}(:, :, :, 2)), ctransform);
      end

      imdist(c) = norm(imdiff(:)) / length(imdiff(:));

      if featdist(c) <= .001 && imdist(c) <= 0.01,
        fprintf('skipping %s: %s %i/%i since they are identical!!!\n', dirs(d).name, files(f).name, f, length(files));
        continue;
      end

      c = c + 1;
    end
  end

  if c > 1,
    plotdata(plotnames(1:c-1), featdist(1:c-1), featratio(1:c-1), imdist(1:c-1), dohog);
  end
end

fprintf('number of points:\n');
plotnames = plotnames(1:c-1);
uplotnames = unique(plotnames);
for i=1:length(uplotnames),
  fprintf('  %s: %i\n', uplotnames{i}, sum(strcmp(plotnames, uplotnames{i})));
end



function plotdata(plotnames, featdist, featratio, imdist, dohog),

clf;

subplot(121);
hold on;

uplotnames = unique(plotnames);
colors = lines(length(uplotnames));

legends = zeros(length(uplotnames), 1);
legendnames = cell(length(uplotnames), 1);

for i=1:length(uplotnames),
  active = strcmp(plotnames, uplotnames{i});

  feat = featdist(active);
  im = imdist(active);

  [~, iii] = sort(feat);

  k = .001;
  smoothfeat = min(feat):k:max(feat);
  smoothim = zeros(size(smoothfeat));
  for j=1:length(smoothfeat),
    jjj = (feat>=smoothfeat(j)).*(feat<smoothfeat(j)+k);
    if length(jjj) > 10,
      smoothim(j) = median(im(logical(jjj)));
    else,
      smoothim(j) = NaN;
    end
  end
  
  legends(i) = plot(smoothfeat, smoothim, '-', 'color', colors(i, :), 'MarkerSize', 20, 'LineWidth', 5);
  plot(feat, im, '*', 'color', colors(i, :), 'MarkerSize', 2);
  legendnames{i} = uplotnames{i};

%  bestfit = polyfit(feat(iii), im(iii), 2);
%  bestfitres = linspace(min(feat), max(feat), 10);
%  plot(bestfitres, polyval(bestfit, bestfitres), '-', 'color', colors(i, :), 'LineWidth', 5);
end

legend(legends, legendnames, 'FontSize', 20);
xlabel('Feat Distance', 'FontSize', 20);

if dohog,
  ylabel('HOG of Image Distance', 'FontSize', 20);
else,
  ylabel('Image Distance', 'FontSize', 20);
end

xlim([0 max(featdist)]);
ylim([0 max(imdist)]);

subplot(122);

lb = 0.8;
ub = 1.2;
dim = 100;
del = [find(featratio < lb) find(featratio > ub)];
featratio(del) = [];
imdist(del) = [];
plotnames(del) = [];

featratio(:) = featratio(:) - lb;
featratio(:) = featratio(:) / (ub - lb);
featratio = ceil(featratio * (dim-1)) + 1;
imdist(:) = imdist(:) - min(imdist(:));
imdist(:) = imdist(:) / max(imdist(:));
imdist = ceil(imdist * (dim-1)) + 1;

allims = {};
for i=1:length(uplotnames),
  active = strcmp(plotnames, uplotnames{i});
  im = zeros(max(imdist), max(featratio));
  ind = sub2ind(size(im), imdist(active), featratio(active));
  t = tabulate(ind);
  im(t(:, 1)) = t(:, 2);
  im = im ./ repmat(max(im)+1, [size(im,1) 1]);
  im = flipud(im);
  allims{i} = im;
end

bigim = montage(allims, 3, 2, 1);

imagesc(bigim);
axis image;
colormap gray;

ptitle = uplotnames{1};
for i=2:length(uplotnames),
  ptitle = [ptitle ', ' uplotnames{i}];
end
title(ptitle, 'FontSize', 20);

drawnow;
