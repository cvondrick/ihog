function [master, chunks] = resolveCNNchunks(masterfile),

fprintf('icnn: locating chunkfiles...\n');
master = load(masterfile);
master = master.master;

chunks = cell(length(master.files), 1);
for i=1:length(chunks),
  chunks{i} = sprintf('%s/%s', fileparts(masterfile), master.files{i});
  fprintf('icnn: found chunkfile %s\n', chunks{i});
end
