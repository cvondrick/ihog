addpath(genpath(pwd));

scandir = '/scratch/hallucination-daemon/images/';
outdir = '/scratch/hallucination-daemon/out/';
maxdim = 500;

fprintf('scan: %s\n', scandir);
fprintf('out: %s\n', outdir);

while true,
  
  files = dir(scandir);

  num = 0;

  for i=1:length(files);
    if files(i).isdir,
      continue;
    end

    filepath = [scandir '/' files(i).name];
    fprintf('found %s', files(i).name);

    t = tic;

    im = imread(filepath);

    scale = 1;
    if size(im, 1) > maxdim,
      scale = maxdim / size(im, 1);
    end
    if size(im, 2) * scale > maxdim,
      scale = maxdim / size(im, 2);
    end

    if scale ~= 1,
      fprintf(', rescale by %f', scale); 
      im = imresize(im, [size(im,1)*scale size(im,2)*scale]);
    end

    im = double(im) / 255.;
    feat = features(im, 8);
    ihog = invertHOG(feat);
    glyph = HOGpicture(feat);

    ihog = imresize(ihog, [size(im,1) size(im,2)]);
    ihog(ihog > 1) = 1;
    ihog(ihog < 0) = 0;
    ihog = repmat(ihog, [1 1 3]);

    glyph = imresize(glyph, [size(im,1) size(im,2)]);
    glyph(glyph > 1) = 1;
    glyph(glyph < 0) = 0;
    glyph = repmat(glyph, [1 1 3]);

    im = padarray(im, [10 10], 0.5);
    ihog = padarray(ihog, [10 10], 0.5);
    glyph = padarray(glyph, [10 10], 0.5);

    vis = [glyph ihog im];

    imwrite(vis, [outdir '/' files(i).name]);

    fprintf(', done in %f.2fs\n', toc(t));

    delete([scandir '/' files(i).name]);

    num = num + 1;
  end

  pause(0.5);
end
