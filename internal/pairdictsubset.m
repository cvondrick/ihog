function pd2 = pairdictsubset(pd, n, mode, ascend),

if ~exist('mode', 'var'),
  mode = 'stdhog';
end

if ~exist('ascend', 'var'),
  ascend = 'ascend';
end

if strcmp(mode, 'stdhog'),
  norms = std(pd.dhog);

elseif strcmp(mode, 'stdgray'),
  norms = std(pd.dgray);

elseif strcmp(mode, 'reds') || strcmp(mode, 'greens') || strcmp(mode, 'blues'),
  if strcmp(mode, 'reds'),
    channel = 1;
  elseif strcmp(mode, 'greens'),
    channel = 2;
  else,
    channel = 3;
  end

  reds = reshape(pd.dgray, [(pd.ny+2)*pd.sbin (pd.nx+2)*pd.sbin 3 size(pd.dgray,2)]);
  reds = reds(:, :, channel, :);
  reds = reshape(reds, [(pd.ny+2)*pd.sbin*(pd.nx+2)*pd.sbin size(pd.dgray,2)]);
  norms = sum(reds);
end

[~,iii] = sort(norms, ascend);
iii = iii(1:n);

pd2 = pd;
pd2.dhog = pd.dhog(:, iii);
pd2.dgray = pd.dgray(:, iii);
