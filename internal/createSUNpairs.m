function createSUNpairs(),

imdir = '/data/vision/torralba/cityimage/SUN_source_code_v2/data/scene_397class/image/SUN397/';
outdir = '/data/vision/torralba/hallucination/ihogj/data';

f = fopen('/data/vision/torralba/hallucination/ihogj/sun.txt', 'r');
data = textscan(f, '%s');
data = data(randperm(length(data)));
data = data{1};
fclose(f);

for i=1:length(data),
  tline = data{i};

  outpath = sprintf('%s/%s.mat', outdir, tline);
  
  if exist(outpath, 'file'),
    continue;
  end

  im = imread([imdir '/' tline]);
  im = imresize(im, [256 256]);

  %if size(im,1) > size(im,2),
  %  im = imresize(im, [500 NaN]);
  %else,
  %  im = imresize(im, [NaN 500]);
  %end

  feat = computeHOG(im2double(im), 8);
  ihog = invertHOG(feat);

  subplot(121);
  imagesc(im); axis image;

  subplot(122);
  imagesc(ihog); axis image;
  colormap gray;
  drawnow;

  [fol, ~] = fileparts(tline);
  if ~exist([outdir '/' fol], 'dir'),
    mkdir([outdir '/' fol]);
  end

  save(outpath, 'feat', 'ihog');
end
