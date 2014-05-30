function exp_reclip(p),

outpath = '/data/vision/torralba/hallucination/icnn/experiments/';

dirs = dir(outpath);
for i=1:length(dirs),
  if ~dirs(i).isdir || dirs(i).name(1) == '.',
    continue;
  end

  if length(strfind(dirs(i).name, 'reclip')),
    continue;
  end

  newdir = sprintf('%s-reclip=%0.2f', dirs(i).name, p);
  mkdir([outpath newdir]);

  files = dir([outpath '/' dirs(i).name]);
  for j=1:length(files),
    if files(j).isdir,
      continue;
    end
    
    payload = load([outpath '/' dirs(i).name '/' files(j).name]);
    for k=1:length(payload.out),
      fprintf('reclipping %s/%s #%i to %0.2f percentile\n', dirs(i).name, files(j).name, k, p);
      payload.out{k} = reclip(payload.out{k}, p);
    end
    save([outpath '/' newdir '/' files(j).name], '-struct', 'payload');
  end
end
