% invertHOGtriangle(feat)
%
% Attempts to reconstruct the image for the HOG features 'feat' using a brute
% force algorithm that repeatedly adds triangles to an image only if doing
% so improves the reconstruction.
%
% Optionally, you can specify an initialization image 'init' to use as
% the starting point. Otherwise, by default it initializes with gray.
function reconstruction = invertHOGtriangle(feat, init, time, draw, sbin),

[ny, nx, ~] = size(feat);

if ~exist('time', 'var'),
  time = 86400;
end
if ~exist('draw', 'var'),
  draw = false;
end
if ~exist('sbin', 'var'),
  sbin = 8;
end

iters = time * 25;

ry = (ny+2)*sbin;
rx = (nx+2)*sbin;

if ~exist('init', 'var'),
  init = 0.5 * ones(ry, rx);
end

core = tril(ones(max(ry, rx)));

objhistory = zeros(iters, 1);
goodtrials = 0;
acceptances = zeros(iters, 1);

reconstruction = init;
changes = zeros(size(init));
objective = -1;

starttime = tic();

for iter=1:iters,
  itertime = toc(starttime);

  fprintf('ihog: iter#%i: timeleft=%0.2fs, rate=%0.2fhz, ', iter, time - itertime, iter / itertime);

  rot = rand() * 360;             % rotate
  w = floor(rand() * sbin*4)+1;   % width
  h = floor(rand() * sbin*4)+1;   % height
  x = floor(rand() * rx);         % center x
  y = floor(rand() * ry);         % center y 
  int = (rand()-0.5) * 0.5;       % intensity

  trial = imrotate(core, rot);
  trial = imresize(trial, [h w]);
  trial = int * trial;

  % calculate position in reconstruction from center of triangle
  ix = floor(x - w/2);
  iy = floor(y - h/2);

  if ix+w > rx,
    trial = trial(:, 1:end-(ix+w-rx));
    w = rx-ix;
  end
  if iy+h > ry,
    trial = trial(1:end-(iy+h-ry), :);
    h = ry-iy;
  end

  if ix < 1,
    trial = trial(:, 1-ix:end);
    w = w-(1-ix)+1;
    ix = 1;
  end
  if iy < 1,
    trial = trial(1-iy:end, :);
    h = h-(1-iy)+1;
    iy = 1;
  end

  candidate = reconstruction;
  candidate(iy:iy+h-1, ix:ix+w-1) = candidate(iy:iy+h-1, ix:ix+w-1) + trial;
  candidate(candidate > 1) = 1;
  candidate(candidate < 0) = 0;

  candidatechanges = changes;
  candidatechanges(iy:iy+h-1, ix:ix+w-1) = candidatechanges(iy:iy+h-1, ix:ix+w-1) + trial;

  candidatefeat = features(repmat(candidate, [1 1 3]), sbin);
  candidateobj = norm(candidatefeat(:) - feat(:), 2) / (ny*nx);

  fprintf('old=%f, new=%f', objective, candidateobj);

  if iter==1 || candidateobj < objective,
    reconstruction = candidate;
    changes = candidatechanges;
    objective = candidateobj;
    goodtrials = goodtrials + 1;
    objhistory(goodtrials) = candidateobj;
    acceptances(iter) = 1;
    fprintf(', accept!');
  else,
    acceptances(iter) = -1;
  end
  fprintf('\n');

  if draw && mod(iter, 1000) == 0,
    subplot(241);
    imagesc(reconstruction); axis image;
    title('Reconstruction');
    subplot(242);
    imagesc(changes); axis image;
    title('Changes');
    subplot(243);
    showHOG(candidatefeat - mean(candidatefeat(:))); axis image;
    title('Reconstruction HOG');
    subplot(244);
    showHOG(feat - mean(feat(:)));
    title('Target HOG');
    subplot(223);
    plot(objhistory(1:goodtrials));
    title('Objective');
    subplot(224);
    accim = zeros(1, iter, 3);
    accim(:, find(acceptances == 1), 2) = 1;
    accim(:, find(acceptances == -1),  1) = 1;
    imagesc(accim);
    title('Acceptances');
    drawnow;
  end

  if itertime > time,
    fprintf('ihog: breaking after %0.2fs\n', itertime);
    break;
  end
end
