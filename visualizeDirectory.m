function visualizeDirectory(dirpath, outpath, pd, ny, nx),

mkdir(outpath);

files = dir(dirpath);
for i=1:length(files),
  if files(i).isdir,
    continue;
  end
  try,
    p = load([dirpath '/' files(i).name]);
  catch
    fprintf('*** unable to load %s\n', files(i).name);
    continue;
  end
  try,
    [inverse, original] = visualizeCNN(p, pd, ny, nx);
  catch
    fprintf('*** unable to process %s\n', files(i).name);
    continue;
  end

  fprintf('loaded %s\n', files(i).name);

  pic = cat(2, inverse, original);

  imagesc(pic);
  axis image;
  drawnow;
  pause;

  imwrite(pic, [outpath '/' files(i).name '.jpg']);
end
