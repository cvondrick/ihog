% graph corresponding to figure 1 in arXiv:1204.4539v1
fprintf('test mexCountConnexComponents\n');
% this graph is a DAG
G=[0 0 0 0 0 0 0 0 0 0 0 0 0;
   1 0 0 0 0 0 0 0 0 0 0 0 0;
   1 1 0 0 0 0 0 0 0 0 0 0 0;
   1 1 0 0 0 0 0 0 0 0 0 0 0;
   0 0 0 1 0 0 0 0 0 0 0 0 0;
   0 1 1 0 1 0 0 0 0 0 0 0 0;
   0 1 0 0 1 0 0 0 0 0 0 0 0;
   0 0 0 0 0 1 1 0 0 0 0 0 0;
   1 0 0 1 0 0 0 0 0 0 0 0 0;
   1 0 0 0 0 0 0 0 1 0 0 0 0;
   0 0 0 0 0 0 0 0 1 1 0 0 0;
   0 0 0 0 0 0 0 0 1 0 1 0 0;
   0 0 0 0 1 0 0 0 1 0 0 1 0];
G=sparse(G);
nodes=[0 1 1 0 0 1 0 0 1 0 1 1 0];
num=mexCountConnexComponents(G,nodes);
fprintf('Num of connected components: %d\n',num);

% this graph is a not a DAG anymore. This function works
% with general graphs.
G=[0 0 0 0 0 0 0 0 0 0 0 0 0;
   1 0 0 0 0 0 0 0 0 0 0 0 0;
   1 1 0 1 0 0 0 0 0 0 0 0 0;
   1 1 0 0 0 0 0 0 0 0 0 0 0;
   0 0 0 1 0 0 0 0 0 0 0 0 0;
   0 1 1 0 1 0 0 0 0 0 0 0 0;
   0 1 0 0 1 0 0 0 0 0 0 0 0;
   0 0 0 0 0 1 1 0 0 0 0 0 0;
   1 0 0 1 0 0 0 0 0 0 0 0 0;
   1 0 0 0 0 0 0 0 1 0 0 0 0;
   0 0 0 0 0 0 0 0 1 1 0 0 0;
   0 0 0 0 0 0 0 0 1 0 1 0 0;
   0 0 0 0 1 0 0 0 1 0 0 1 0];
nodes=[0 1 1 0 0 1 0 0 1 0 1 1 0];
G=sparse(G);
num=mexCountConnexComponents(G,nodes);
fprintf('Num of connected components: %d\n',num);

nodes=[0 1 1 1 0 1 0 0 1 0 1 1 0];
num=mexCountConnexComponents(G,nodes);
fprintf('Num of connected components: %d\n',num);
