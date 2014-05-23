function [mu, sig] = computeCNNmean(dirpath, maxn)

mu = zeros(1, 9216);
sig = 0;
n = 0;

files = dir(dirpath);
files = files(randperm(length(files)));
for i=1:length(files),
  if files(i).isdir,
    continue;
  end
  if ~strcmp(files(i).name(end-3:end), '.mat'),
    continue;
  end

  load([dirpath '/' files(i).name]);

  fprintf('load %s (%i of %i)\n', files(i).name, n, maxn);

  mu = mu + sum(feat, 1);
  sig = sig + sum(feat(:).^2);
  n = n + size(feat, 2);

  if n > maxn,
    break;
  end
end

mu = mu / n;
sig = sig / (n*9216) - mean(mu)^2;
