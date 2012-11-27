clear all;
rand('seed',0);
randn('seed',0);
fprintf('test mexFistaPathCoding\n');
p=100;
n=1000;
% generate a DAG
G=sprand(p,p,0.05);
G=mexRemoveCyclesGraph(G);
fprintf('\n');

% generate a data matrix
X=randn(n,p);
X=X-repmat(mean(X),[size(X,1) 1]);
X=mexNormalize(X);
Y=randn(n,2);
Y=Y-repmat(mean(Y),[size(Y,1) 1]);
Y=mexNormalize(Y);
W0=zeros(size(X,2),size(Y,2));

% input graph
graph.weights=G;
graph.stop_weights=zeros(1,p);
graph.start_weights=10*ones(1,p);

% FISTA parameters
param.num_threads=-1; % all cores (-1 by default)
param.verbose=true;   % verbosity, false by default
param.lambda=0.005; % regularization parameter
param.it0=1;      % frequency for duality gap computations
param.max_it=100; % maximum number of iterations
param.L0=0.01;
param.tol=1e-4;
param.precision=10000000;
param.pos=false;

fprintf('Square Loss + convex path penalty\n');
param.loss='square';
param.regul='graph-path-conv';
tic
[W1 optim_info]=mexFistaPathCoding(Y,X,W0,graph,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
num=mexCountConnexComponents(graph.weights,W1(:,1));
fprintf('Num of connected components: %d\n',num);

fprintf('\n');
fprintf('Square Loss + non-convex path penalty\n');
param.loss='square';
param.regul='graph-path-l0';
param.lambda=0.0001; % regularization parameter
param.ista=true;
tic
[W2 optim_info]=mexFistaPathCoding(Y,X,W0,graph,param);
t=toc;
num=mexCountConnexComponents(graph.weights,W2(:,1));
fprintf('Num of connected components: %d\n',num);

fprintf('\n');
fprintf('Note that for non-convex penalties, continuation strategies sometimes perform better:\n');
tablambda=param.lambda*sqrt(sqrt(sqrt(2))).^(20:-1:0);
lambda_orig=param.lambda;
tic
W2=W0;
for ii = 1:length(tablambda)
   param.lambda=tablambda(ii);
   param.verbose=false;
   [W2]=mexFistaPathCoding(Y,X,W2,graph,param);
end
param.verbose=true;
param.lambda=lambda_orig;
[W2 optim_info]=mexFistaPathCoding(Y,X,W2,graph,param);
t=toc;
num=mexCountConnexComponents(graph.weights,W2(:,1));
param.ista=false;
fprintf('Num of connected components: %d\n',num);






