% pd2txt(out, pd)
%
% Dumps the paired dictionary 'pd' to output file 'file'
function pd2txt(out, pd),

f = fopen(out, 'w');
fprintf(f, '%f\n', pd.lambda);
fprintf(f, '%i %i\n', pd.ny, pd.nx);
fprintf(f, '%i\n', pd.sbin);
fprintf(f, '%i\n', pd.k);
fprintf(f, '\n');
for i=1:pd.k,
  fprintf(f, '%f ', pd.dhog(i, :));
  fprintf(f, '\n');
end
fprintf(f, '\n');
for i=1:pd.k,
  fprintf(f, '%f ', pd.dgray(i, :));
  fprintf(f, '\n');
end
fclose(f);
