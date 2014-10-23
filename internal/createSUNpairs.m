function createSUNpairs(),

seedrandom();

imdir = '/data/vision/torralba/cityimage/SUN_source_code_v2/data/scene_397class/image/SUN397/';
outdir = '/data/vision/torralba/hallucination/ihogj/data';

f = fopen('/data/vision/torralba/hallucination/ihogj/sun.txt', 'r');
data = textscan(f, '%s');
data = data{1};
data = data(randperm(length(data)));
fclose(f);

for i=1:length(data),
  tline = data{i};

  outpath = sprintf('%s/%s.mat', outdir, tline);
  
  if exist(outpath, 'file'),
    continue;
  end

  fprintf('%s\n', outpath);

  im = imread([imdir '/' tline]);
  if size(im,3) == 1,
    im = repmat(im, [1 1 3]);
  end
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



% Generates a random seed for MATLAB that is robust against many problems that
% crop up when laucning jobs on the cluster. It is better than just seeding
% with the clock since cluster jobs may start at the *exact* same time.
function seed = seedrandom(),

[~, hostname] = system('hostname');
hostname = strtrim(hostname);
hostname = double(hostname);
hostname = sum(hostname);

[~, randnum] = system('echo $RANDOM');
randnum = strtrim(randnum);
randnum = str2num(randnum);

pid = feature('getpid');

seed = hostname * randnum * pid;
seed = mod(seed, 2^31);

rng(seed);

fprintf('random seed set to %i = %i * %i * %i\n', seed, hostname, randnum, pid);
