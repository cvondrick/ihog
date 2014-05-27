function buildCNNchunks(sourcedir, outdir, chunksize, hogdim),

if ~exist('chunksize', 'var'),
  chunksize = 100000;
end
if ~exist('featdim', 'var'),
  featdim = 9216;
end
if ~exist('imagedim', 'var'),
  imagedim = 64;
end
if ~exist('hogdim', 'var'),
  hogdim = [0 0];
end

data = zeros(prod(featdim)+imagedim^2*3+prod(hogdim)*computeHOG(), chunksize, 'single');
c = 1;
chunkid = 1;
n = 0;

fprintf('cnn: initializing image mean\n');
input_size = 227;
mean_image_file = '/data/vision/torralba/hallucination/caffe/ilsvrc_2012_mean.mat';
image_mean = load(mean_image_file);
image_mean = image_mean.image_mean;
off = floor((size(image_mean,1) - input_size)/2)+1;
image_mean = image_mean(off:off+input_size-1, off:off+input_size-1, :);

master.imdim = [imagedim imagedim 3];
master.featdim = featdim;
master.hogdim = hogdim;

warning off; mkdir(outdir); warning on;

files = dir(sourcedir);
files = files(randperm(length(files)));
for i=1:length(files),
  if files(i).isdir,
    continue;
  end
  if ~strcmp(files(i).name(end-3:end), '.mat'),
    continue;
  end

  fprintf('icnn: load: %s (chunk has %i/%i or %.1f%%)\n', files(i).name, c, chunksize, c / chunksize * 100);

  [~, basename] = fileparts(files(i).name);
  imfull = double(imread([sourcedir '/images/' basename '.jpg']));
  imfullpad = padarray(imfull, [8 8], 0);

  payload = load([sourcedir '/' files(i).name]);

  num = size(payload.feat,1);
  n = n + num;

  for j=1:num,
    im = im_crop(imfull, payload.boxes(j, :), 'warp', size(image_mean,1), 16, image_mean);
    im = imresize(im, [imagedim imagedim]);
    im = single(im) / 255.;

    feat = payload.feat(j, :);

    % normalize
    im(:) = im(:) - mean(im(:));
    im(:) = im(:) / (sqrt(sum(im(:).^2) + eps));

    feat(:) = feat(:) - mean(feat(:));
    feat(:) = feat(:) / (sqrt(sum(feat(:).^2) + eps));

    data(1:imagedim^2*3, c) = im(:);
    data(imagedim^2*3+1:imagedim^2*3+prod(featdim), c) = feat(:);

    if prod(hogdim) > 0,
      xmin = payload.boxes(j, 1);
      ymin = payload.boxes(j, 2);
      xmax = payload.boxes(j, 3);
      ymax = payload.boxes(j, 4);

      imcrop = imfullpad(ymin:ymax+16, xmin:xmax+16, :);

      hogpatch = double(imresize(imcrop, (hogdim+2)*8));
      hog = computeHOG(hogpatch, 8);
      hog(:) = hog(:) - mean(hog(:));
      hog(:) = hog(:) / (sqrt(sum(hog(:).^2) + eps));
      data(imagedim^2*3+prod(featdim)+1:end, c) = hog(:);
    end

    c = c + 1;

    if c > chunksize,
      outpath = sprintf('%s/chunk-%i.mat', outdir, chunkid);
      fprintf('icnn: flush chunk %i to %s\n', chunkid, outpath);
      master.files{chunkid} = sprintf('chunk-%i.mat', chunkid);
      save(outpath, '-v7.3', 'data');
      c = 1;
      chunkid = chunkid + 1;

      master.n = n;
      save(sprintf('%s/master.mat', outdir), '-v7.3', 'master');
    end
  end
end

data = data(:, 1:c-1);
master.files{chunkid} = sprintf('chunk-%i.mat', chunkid);
fprintf('icnn: flush chunk %i\n', chunkid);
save(sprintf('%s/chunk-%i.mat', outdir, chunkid), '-v7.3', 'data');

master.n = n;
save(sprintf('%s/master.mat', outdir), '-v7.3', 'master');




function window = im_crop(im, bbox, crop_mode, crop_size, padding, image_mean)
% window = rcnn_im_crop(im, bbox, crop_mode, crop_size, padding, image_mean)
%   Crops a window specified by bbox (in [x1 y1 x2 y2] order) out of im.
%
%   crop_mode can be either 'warp' or 'square'
%   crop_size determines the size of the output window: crop_size x crop_size
%   padding is the amount of padding to include at the target scale
%   image_mean to subtract from the cropped window
%
%   N.B. this should be as identical as possible to the cropping 
%   implementation in Caffe's WindowDataLayer, which is used while
%   fine-tuning.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

use_square = false;
if strcmp(crop_mode, 'square')
  use_square = true;
end

% defaults if padding is 0
pad_w = 0;
pad_h = 0;
crop_width = crop_size;
crop_height = crop_size;
if padding > 0 || use_square
  %figure(1); showboxesc(im/256, bbox, 'b', '-');
  scale = crop_size/(crop_size - padding*2);
  half_height = (bbox(4)-bbox(2)+1)/2;
  half_width = (bbox(3)-bbox(1)+1)/2;
  center = [bbox(1)+half_width bbox(2)+half_height];
  if use_square
    % make the box a tight square
    if half_height > half_width
      half_width = half_height;
    else
      half_height = half_width;
    end
  end
  bbox = round([center center] + ...
      [-half_width -half_height half_width half_height]*scale);
  unclipped_height = bbox(4)-bbox(2)+1;
  unclipped_width = bbox(3)-bbox(1)+1;
  %figure(1); showboxesc([], bbox, 'r', '-');
  pad_x1 = max(0, 1 - bbox(1));
  pad_y1 = max(0, 1 - bbox(2));
  % clipped bbox
  bbox(1) = max(1, bbox(1));
  bbox(2) = max(1, bbox(2));
  bbox(3) = min(size(im,2), bbox(3));
  bbox(4) = min(size(im,1), bbox(4));
  clipped_height = bbox(4)-bbox(2)+1;
  clipped_width = bbox(3)-bbox(1)+1;
  scale_x = crop_size/unclipped_width;
  scale_y = crop_size/unclipped_height;
  crop_width = round(clipped_width*scale_x);
  crop_height = round(clipped_height*scale_y);
  pad_x1 = round(pad_x1*scale_x);
  pad_y1 = round(pad_y1*scale_y);

  pad_h = pad_y1;
  pad_w = pad_x1;

  if pad_y1 + crop_height > crop_size
    crop_height = crop_size - pad_y1;
  end
  if pad_x1 + crop_width > crop_size
    crop_width = crop_size - pad_x1;
  end
end % padding > 0 || square

window = im(bbox(2):bbox(4), bbox(1):bbox(3), :);
% We turn off antialiasing to better match OpenCV's bilinear 
% interpolation that is used in Caffe's WindowDataLayer.
tmp = imresize(window, [crop_height crop_width], ...
    'bilinear', 'antialiasing', false);
if ~isempty(image_mean)
  tmp = tmp - image_mean(pad_h+(1:crop_height), pad_w+(1:crop_width), :);
end
%figure(2); window_ = tmp; imagesc((window_-min(window_(:)))/(max(window_(:))-min(window_(:)))); axis image;
window = zeros(crop_size, crop_size, 3, 'single');
window(pad_h+(1:crop_height), pad_w+(1:crop_width), :) = tmp;
%figure(3); imagesc((window-min(window(:)))/(max(window(:))-min(window(:)))); axis image; pause;
