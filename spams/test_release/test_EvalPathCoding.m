clear all;
rand('seed',0);
randn('seed',0);

p=100;
G=sprand(p,p,0.05);
G=mexRemoveCyclesGraph(G);
fprintf('\n');

% input graph
graph.weights=G;
graph.stop_weights=zeros(1,p);
graph.start_weights=10*ones(1,p);

param.regul='graph-path-l0';
U=randn(p,10);
U=U-mean(U(:));
U=mexNormalize(U);
param.lambda=0.005;
[V2 optim]=mexProximalPathCoding(U,graph,param);
[vals paths]=mexEvalPathCoding(U,graph,param);
