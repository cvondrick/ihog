function resp = visualize(),



category = 'person';
base = '~/voc-release5/person_parts_full/';
[filename, a, b, s] = textread('person.txt', '%s %d %d %f');
%
%category = 'chair';
%base = '~/voc-release5/chair_parts_25/';
%[filename, a, b, s] = textread('report-chair-dpm.txt', '%s %d %d %f');
%%%%
category = 'car';
base = '~/voc-release5/car_parts/';
[filename, a, b, s] = textread('car.txt', '%s %d %d %f');
%%
%category = 'cat';
%base = '~/voc-release5/cat_parts/';
%[filename, a, b, s] = textread('cat.txt', '%s %d %d %f');

%category = 'chair_rgb';
%base = '~/voc-release5/chair_parts_25/';
%[filename, a, b, s] = textread('chair_rgb.txt', '%s %d %d %f');
%

mkdir(sprintf('dump/%s/ihog', category));
mkdir(sprintf('dump/%s/ihog/tp', category));
mkdir(sprintf('dump/%s/ihog/fp', category));
mkdir(sprintf('dump/%s/original/tp', category));
mkdir(sprintf('dump/%s/original/fp', category));

keeps = zeros(length(filename), 1);
for i=1:length(filename),
  data = load(sprintf('%s/mats/%s.mat', base, filename{i}));
  if data.isgood,
    keeps(i) = 1;
  elseif data.overlap <= 0,
    keeps(i) = 1;
  end
end

fprintf('keeping %i of %i\n', sum(keeps), length(keeps));

filename = filename(keeps == 1);
s = s(keeps == 1);

boundaries = 10; % how many to look around the boundaries

detscores = zeros(length(filename), 1);

goodstr = zeros(length(filename),1);
numpos = 0;
for i=1:length(filename),
  data = load(sprintf('%s/mats/%s.mat', base, filename{i}));
  if data.isgood,
    goodstr(i) = 1;
    numpos = numpos + 1;
  end
  detscores(i) = data.bbox(end);
end

[~, dpmiii] = sort(detscores, 'descend');

tpDet=zeros(length(filename),1);
fpDet=zeros(length(filename),1);
for i=1:length(filename),
  data = load(sprintf('%s/mats/%s.mat', base, filename{dpmiii(i)}));
  if data.isgood,
    tpDet(i) = 1;
  else,
    fpDet(i) = 1;
  end
end

% compute precision/recall
fpCdet=cumsum(fpDet(1:i));
tpCdet=cumsum(tpDet(1:i));
recDet=tpCdet/numpos;
precDet=tpCdet./(fpCdet+tpCdet);
apDet = VOCap(recDet, precDet);

tp=zeros(length(filename),1);
fp=zeros(length(filename),1);

%s = s + 0.0001 * goodstr;
%s = s + 0.00001 * rand(length(s), 1);

[~, iii] = sort(s, 'descend');
%[~, iii] = sort(detscores, 'ascend');

figure(1);
clf;
percentoverlap = zeros(length(filename),1);
random = randperm(length(filename));
for i=2:length(filename),
  dpmtop = dpmiii(1:i);
  hogglestop = iii(1:i);
  corr = corrcoef(dpmtop, hogglestop);
  percentoverlap(i) = corr(1,2);
end
xxx = (1:length(filename))/length(filename);
xxx = xxx(2:end);
percentoverlap = percentoverlap(2:end);

plot(xxx, percentoverlap, 'LineWidth', 2);
hold on;
plot([0 1], [0 0], 'k--', 'LineWidth', 2);
xlabel('Window Rank');
ylabel('Percent Overlap');
ylim([-1 1]);
xlim([0 1]);

resp.percentoverlap = percentoverlap;
resp.xxx = xxx;





topim = [];
topheight = 200;
topcount = 0;
topcountgood = 0;
topcountgoodmax = 3;

figure(1);

dpmscores = zeros(length(filename), 1);
ihogscores = zeros(length(filename), 1);
goodscores = zeros(length(filename), 1);
recall = 0;
for i=1:length(filename),
%for i=[4 59],
  fprintf('%s: %f', filename{iii(i)}, s(iii(i)));

  data = load(sprintf('%s/mats/%s.mat', base, filename{iii(i)}));

  orig = imread(sprintf('%s/original/%s', base, filename{iii(i)}));
  ihog = imread(sprintf('%s/ihog/%s', base, filename{iii(i)}));
  subplot(331);
  imagesc(imread(sprintf('%s/originalfull/%s', base, filename{iii(i)})));
  axis image;
  title(sprintf('good: %i', data.isgood), 'FontSize', 20);

  subplot(332);
  imagesc(orig);
  axis image;
  title(sprintf('score: %f', data.bbox(end)), 'FontSize', 20);


  subplot(333);
  imagesc(ihog);
  axis image;
  title(sprintf('hoggles score: %f', s(iii(i))), 'FontSize', 20);
  colormap gray;

  dpmscores(i) = data.bbox(end);
  ihogscores(i) = s(iii(i));
  goodscores(i) = data.isgood;

  if data.isgood,
    tp(i) = 1;
  else,
    fp(i) = 1; 
  end

  if data.isgood,
    folder = 'tp';
  else,
    folder = 'fp';
  end

  %imwrite(ihog, sprintf('dump/%s/ihog/%s/%07i.jpg', category, folder, i));
  %imwrite(orig, sprintf('dump/%s/original/%s/%07i.jpg', category, folder, i));

  % compute precision/recall
  fpC=cumsum(fp(1:i));
  tpC=cumsum(tp(1:i));
  rec=tpC/numpos;
  prec=tpC./(fpC+tpC);

  subplot(334);
  scatter(dpmscores(logical(goodscores(1:i))), ihogscores(logical(goodscores(1:i))), 'bo');
  xlim([min(detscores) max(detscores)]);
  ylim([min(s) max(s)]);
  xlabel('DT Score');
  ylabel('Hoggles Score');

  subplot(335);
  scatter(dpmscores(~logical(goodscores(1:i))), ihogscores(~logical(goodscores(1:i))), 'bo');
  xlim([min(detscores) max(detscores)]);
  ylim([min(s) max(s)]);
  xlabel('DT Score');
  ylabel('Hoggles Score');

  subplot(336);
  cla;
  plot(rec,prec,'-');
  hold on;
  plot(recDet, precDet, 'r-');
  plot([0 1], [numpos / length(filename), numpos / length(filename)], 'k--');
  ap = VOCap(rec, prec);
  title(sprintf('AP = %f (DPM = %f)', ap, apDet), 'FontSize', 20);
  ylim([0, 1]);
  xlim([0, 1]);

  if i > 0,
    if topcount < 20,
      if data.isgood, % (data.isgood && topcountgood < topcountgoodmax) || ~data.isgood, 
        orig = imread(sprintf('%s/original/%s', base, filename{iii(i)}));
        ihog = imread(sprintf('%s/ihog/%s', base, filename{iii(i)}));
        ihog = imresize(ihog, topheight / size(ihog, 1));
        orig = imresize(orig, [size(ihog,1), size(ihog, 2)]);
        ihog = repmat(ihog, [1 1 3]);
        im = [ihog; orig];

        bord = 5;
        im = padarray(im, [bord bord], 255);
        if data.isgood,
          im(:, [1:bord end-bord:end], [1 3]) = 0;
          im([1:bord end-bord:end], :, [1 3]) = 0;
        else,
          im(:, [1:bord end-bord:end], [2 3]) = 0;
          im([1:bord end-bord:end], :, [2 3]) = 0;
        end
        im = padarray(im, [bord bord], 255);

        topim = [topim im];
        topcount = topcount + 1;

        subplot(313);
        imagesc(topim);
        axis image;

        if data.isgood,
          topcountgood = topcountgood + 1;
        end
      end
    else,
      resp.topim = topim;
    end
  end



  if ~data.isgood,
    drawnow;
    pause;
  end


  %car_image4531_box2.jpg
  %car_image2867_box3.jpg
  %car_image418_box1.jpg
  
  %r = corr(dpmscores(1:i), ihogscores(1:i), 'type', 'Spearman');
  %fprintf(', rankcorr = %f\n', r);
  fprintf('\n');

end

rankcor = corr(dpmscores, ihogscores, 'type', 'Spearman')

resp.topim = topim;
resp.rankcor = rankcor;
resp.hoggles.prec = prec;
resp.hoggles.rec = rec;
resp.hoggles.ap = ap;
resp.dpm.prec = precDet;
resp.dpm.rec = recDet;
resp.dpm.ap = apDet; 

figure(2);
plot(rec, prec, 'b-');
hold on;
plot(recDet, precDet, 'r-');
legend(sprintf('Hoggles AP = %0.2f', ap), sprintf('DPM AP = %0.2f', apDet));
xlim([0 1]);
ylim([0 1]);
xlabel('Recall');
ylabel('Precision');
title(sprintf('Chair'));


function ap = VOCap(rec,prec)

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));

