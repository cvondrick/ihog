fprintf('test mexRemoveCyclesGraph\n');
% this graph is not a DAG
G=[0   0   0   0   0   0   0   0   0   0   0   0   0;
   1   0   0   0.1 0   0   0   0.1 0   0   0   0   0;
   1   1   0   0   0   0.1 0   0   0   0   0   0   0;
   1   1   0   0   0   0   0   0   0   0   0   0.1 0;
   0   0   0   1   0   0   0   0   0   0   0   0   0;
   0   1   1   0   1   0   0   0   0   0   0   0   0;
   0   1   0   0   1   0   0   0   0   0   0   0   0;
   0   0   0   0   0   1   1   0   0   0   0   0   0;
   1   0   0   1   0   0   0   0   0   0   0   0   0;
   1   0   0   0   0   0   0   0   1   0   0   0   0;
   0   0   0   0   0   0   0   0   1   1   0   0   0;
   0   0   0   0   0   0   0   0   1   0   1   0   0;
   0   0   0   0   1   0   0   0   1   0   0   1   0];
G=sparse(G);
DAG=mexRemoveCyclesGraph(G);
format compact;
fprintf('Original graph:\n');
full(G)
fprintf('New graph:\n');
full(DAG)
