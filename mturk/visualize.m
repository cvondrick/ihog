function visualize(),

[filename, ~, ~, s] = textread('report.txt', '%s %d %d %f');

%detscores = zeros(length(filename), 1);
%for i=1:length(filename),
%  data = load(sprintf('~/voc-release5/hoggles/mats/%s.mat', filename{i}));
%  detscores(i) = data.bbox(end);
%end
%
%[ssss, iii] = sort(detscores, 'descend');
%
%clf;
%plot(ssss);
%pause;

[~, iii] = sort(s, 'descend');


dpmscores = zeros(length(filename), 1);
ihogscores = zeros(length(filename), 1);


for i=1:length(filename),
  fprintf('%s: %f\n', filename{iii(i)}, s(iii(i)));

  data = load(sprintf('~/voc-release5/hoggles/mats/%s.mat', filename{iii(i)}));

  subplot(131);
  imagesc(imread(sprintf('~/voc-release5/hoggles/originalfull/%s', filename{iii(i)})));
  axis image;
  title(sprintf('good: %i', data.isgood), 'FontSize', 20);

  subplot(132);
  imagesc(imread(sprintf('~/voc-release5/hoggles/original/%s', filename{iii(i)})));
  axis image;
  title(sprintf('score: %f', data.bbox(end)), 'FontSize', 20);

  subplot(133);
  imagesc(imread(sprintf('~/voc-release5/hoggles/ihog/%s', filename{iii(i)})));
  axis image;
  title(sprintf('hoggles score: %f', s(iii(i))), 'FontSize', 20);
  colormap gray;

  pause;

  dpmscores(i) = data.bbox(end);
  ihogscores(i) = s(iii(i));
end

