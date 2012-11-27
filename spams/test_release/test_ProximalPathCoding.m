clear all;
rand('seed',0);
randn('seed',0);
fprintf('test mexProximalPathCoding\n');
p=100;
% generate a DAG
G=sprand(p,p,0.02);
G=mexRemoveCyclesGraph(G);
fprintf('\n');

% generate a data matrix
U=randn(p,10);
U=U-mean(U(:));
U=mexNormalize(U);

% input graph
graph.weights=G;
graph.stop_weights=zeros(1,p);
graph.start_weights=10*ones(1,p);

% FISTA parameters
param.num_threads=-1; % all cores (-1 by default)
param.verbose=true;   % verbosity, false by default
param.lambda=0.05; % regularization parameter

fprintf('Proximal convex path penalty\n');
param.regul='graph-path-conv';
tic
[V1 optim]=mexProximalPathCoding(U,graph,param);
t=toc;
num=mexCountConnexComponents(graph.weights,V1(:,1));
fprintf('Num of connected components: %d\n',num);

fprintf('Proximal non-convex path penalty\n');
param.regul='graph-path-l0';
param.lambda=0.005;
tic
[V2 optim]=mexProximalPathCoding(U,graph,param);
t=toc;
num=mexCountConnexComponents(graph.weights,V2(:,1));
fprintf('Num of connected components: %d\n',num);

graph.start_weights=1*ones(1,p);
param.lambda=0.05;
fprintf('Proximal convex path penalty\n');
param.regul='graph-path-conv';
tic
[V1 optim]=mexProximalPathCoding(U,graph,param);
t=toc;
num=mexCountConnexComponents(graph.weights,V1(:,1));
fprintf('Num of connected components: %d\n',num);

fprintf('Proximal non-convex path penalty\n');
param.regul='graph-path-l0';
param.lambda=0.005;
tic
[V2 optim]=mexProximalPathCoding(U,graph,param);
t=toc;
num=mexCountConnexComponents(graph.weights,V2(:,1));
fprintf('Num of connected components: %d\n',num);







