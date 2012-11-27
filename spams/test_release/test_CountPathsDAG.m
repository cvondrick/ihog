% graph corresponding to figure 1 in arXiv:1204.4539v1
fprintf('test mexCountPathsDAG\n');
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
num=mexCountPathsDAG(G);
fprintf('Num of paths: %d\n',num);
