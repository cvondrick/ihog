function exp_results(dirpath),

samplesize = 100;
mode = 'standard';
dtind = 15;
reclip = false;

dt = train_dt();
dt = dt.w(:, dtind);

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

  if (length(strfind(dirs(d).name, 'reclip=0.02')) == 0) == reclip,
    fprintf('skip %s\n', dirs(d).name);
    continue;
  end

  if length(strfind(dirs(d).name, 'standard')) == 0 && length(strfind(dirs(d).name, 'rgb')) == 0 && length(strfind(dirs(d).name, 'edge')) == 0 && length(strfind(dirs(d).name, 'baseline')) == 0 && length(strfind(dirs(d).name, 'delete')) == 0,
    continue;
  end

  pos = strfind(dirs(d).name, '_');
  if isempty(pos),
    base = dirs(d).name;
  else,
    base = dirs(d).name(1:pos-1);
  end

  files = dir([dirpath '/' dirs(d).name '/feat']);
  if ~strcmp(base, 'delete'),
    if length(files) > samplesize,
      files = files(randperm(length(files), samplesize));
    end
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
    data{f} = payload;

    if strcmp(mode, 'dt'),
      origpayload = load(payload.infile);
      keep = find(origpayload.class == dtind);

      payload.feat = payload.feat(keep);
      payload.orig = payload.orig(keep);
      payload.boxes = payload.boxes(keep);
      payload.out = payload.out(keep);
      payload.refeat = payload.refeat(keep);
    end
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

      if strcmp(mode, 'hog'), 
        imdiff = computeHOG(double(data{f}.out{k}(:, :, :, 1)), 4) - computeHOG(double(data{f}.out{k}(:, :, :, 2)), 4);
        imdiff = norm(imdiff(:)) / length(imdiff(:));
      elseif strcmp(mode, 'dt'), 
        firstim = computeHOG(double(data{f}.out{k}(:, :, :, 1)), 8);
        secondim = computeHOG(double(data{f}.out{k}(:, :, :, 2)), 8);
        imdiff = abs(dt' * (secondim(:) - firstim(:)));
      else,
        imdiff = applycform(double(data{f}.out{k}(:, :, :, 1)), ctransform) - applycform(double(data{f}.out{k}(:, :, :, 2)), ctransform);
        imdiff = norm(imdiff(:)) / length(imdiff(:));
      end

      imdist(c) = imdiff;

      if featdist(c) == 0 && imdist(c) == 0,
        fprintf('skipping %s: %s %i/%i since they are identical!!!\n', dirs(d).name, files(f).name, f, length(files));
        continue;
      end

      c = c + 1;
    end
  end

  if c > 1,
    plotdata(plotnames(1:c-1), featdist(1:c-1), featratio(1:c-1), imdist(1:c-1), mode);
  end
end

fprintf('number of points:\n');
plotnames = plotnames(1:c-1);
uplotnames = unique(plotnames);
for i=1:length(uplotnames),
  fprintf('  %s: %i\n', uplotnames{i}, sum(strcmp(plotnames, uplotnames{i})));
end



function plotdata(plotnames, featdist, featratio, imdist, mode),

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
    if length(jjj) > 25,
      smoothim(j) = median(im(logical(jjj)));
    else,
      smoothim(j) = NaN;
    end
  end
  
  legends(i) = plot(smoothfeat, smoothim, '-', 'color', colors(i, :), 'MarkerSize', 20, 'LineWidth', 5);
  plot(feat, im, '*', 'color', colors(i, :), 'MarkerSize', 1);
  legendnames{i} = uplotnames{i};

%  bestfit = polyfit(feat(iii), im(iii), 2);
%  bestfitres = linspace(min(feat), max(feat), 10);
%  plot(bestfitres, polyval(bestfit, bestfitres), '-', 'color', colors(i, :), 'LineWidth', 5);
end

for i=1:length(legendnames),
  if strcmp(legendnames{i}, 'standard'),
    legendnames{i} = 'Identity (us)';
  elseif strcmp(legendnames{i}, 'rgb'),
    legendnames{i} = 'Color (us)';
  elseif strcmp(legendnames{i}, 'edge'),
    legendnames{i} = 'Edge (us)';
  elseif strcmp(legendnames{i}, 'baseline'),
    legendnames{i} = 'Baseline A';
  elseif strcmp(legendnames{i}, 'delete'),
    legendnames{i} = 'Baseline B';
  end
end

[~, iii] = sort(legendnames);

legend(legends(iii), legendnames(iii), 'FontSize', 20);
xlabel('CNN Distance', 'FontSize', 20);

if strcmp(mode, 'hog'),
  ylabel('HOG of Image Distance', 'FontSize', 20);
elseif strcmp(mode, 'dt'),
  ylabel('DT Score Distance', 'FontSize', 20);
else,
  ylabel('Image Distance', 'FontSize', 20);
end

xlim([0 max(featdist)]);
xlim([0 0.02]);
%ylim([0 max(imdist)]);


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
  im = imresize(im, 4, 'nearest');
  allims{i} = 1-im;

  imwrite(allims{i}, sprintf('~/papers/multiple/figs/ratio_%s.jpg', uplotnames{i}));
end

bigim = montage(allims, 3, 2, 1);

imagesc(bigim, [0 1]);
axis image;
colormap gray;

ptitle = uplotnames{1};
for i=2:length(uplotnames),
  ptitle = [ptitle ', ' uplotnames{i}];
end
title(ptitle, 'FontSize', 20);

drawnow;
