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

end

[~,iii] = sort(norms, ascend);
iii = iii(1:n);

pd2 = pd;
pd2.dhog = pd.dhog(:, iii);
pd2.dgray = pd.dgray(:, iii);
