clear all;
randn('seed',0);
format compact;
param.num_threads=-1; % all cores (-1 by default)
param.verbose=false;   % verbosity, false by default
param.lambda=0.1; % regularization parameter
param.it0=1;      % frequency for duality gap computations
param.max_it=100; % maximum number of iterations
param.L0=0.1;
param.tol=1e-5;
param.intercept=false;
param.pos=false;

graph.eta_g=[1 1 1 1 1];
graph.groups=sparse([0 0 0 1 0;
                     0 0 0 0 0;
                     0 0 0 0 0;
                     0 0 0 0 0;
                     0 0 1 0 0]);   % g5 is included in g3, and g2 is included in g4
graph.groups_var=sparse([1 0 0 0 0; 
                         1 0 0 0 0; 
                         1 0 0 0 0 ; 
                         1 1 0 0 0; 
                         0 1 0 1 0;
                         0 1 0 1 0;
                         0 1 0 0 1;
                         0 0 0 0 1;
                         0 0 0 0 1;
                         0 0 1 0 0]); % represents direct inclusion relations 

X=randn(100,10);
param.verbose=true;
%X=eye(10);
X=X-repmat(mean(X),[size(X,1) 1]);
X=mexNormalize(X);
Y=randn(100,1);
Y=Y-repmat(mean(Y),[size(Y,1) 1]);
Y=mexNormalize(Y);
W0=zeros(size(X,2),size(Y,2));
% Regression experiments 
% 100 regression problems with the same design matrix X.
fprintf('\nVarious regression experiments\n');
param.compute_gram=true;
fprintf('\nFISTA + Regression graph\n');
param.loss='square';
param.regul='graph';
tic
[W optim_info]=mexFistaGraph(Y,X,W0,graph,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

fprintf('\nADMM + Regression graph\n');
param.admm=true;
param.lin_admm=true;
param.c=1;
param.delta=1;
tic
[W optim_info]=mexFistaGraph(Y,X,W0,graph,param);
t=toc;
fprintf('mean loss: %f, stopping criterion: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

param.admm=false;
param.max_it=5;
param.it0=1;
tic
[W optim_info]=mexFistaGraph(Y,X,W,graph,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

% works also with non graph-structured regularization. graph is ignored
fprintf('\nFISTA + Regression Fused-Lasso\n');
param.regul='fused-lasso';
param.lambda2=0.01;
param.lambda3=0.01; %
tic
[W optim_info]=mexFistaGraph(Y,X,W0,graph,param);
t=toc;
fprintf('mean loss: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Regression graph with intercept \n');
param.intercept=true;
param.regul='graph';
tic
[W optim_info]=mexFistaGraph(Y,[X ones(size(X,1),1)],[W0; zeros(1,size(W0,2))],graph,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
param.intercept=false;

% Classification
fprintf('\nOne classification experiment\n');
Y=2*double(randn(100,size(Y,2)) > 0)-1;
fprintf('\nFISTA + Logistic + graph-linf\n');
param.regul='graph';
param.loss='logistic';
param.lambda=0.01;
tic
[W optim_info]=mexFistaGraph(Y,X,W0,graph,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
% can be used of course with other regularization functions, intercept,...

% Multi-Class classification
Y=double(ceil(5*rand(100,size(Y,2)))-1); 
param.loss='multi-logistic';
param.regul='graph';
fprintf('\nFISTA + Multi-Class Logistic + graph \n');
tic
nclasses=max(Y(:))+1;
W0=zeros(size(X,2),nclasses*size(Y,2));
[W optim_info]=mexFistaGraph(Y,X,W0,graph,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
% can be used of course with other regularization functions, intercept,...

% Multi-Task regression
Y=randn(100,size(Y,2));
Y=Y-repmat(mean(Y),[size(Y,1) 1]);
Y=mexNormalize(Y);
param.compute_gram=false;
param.verbose=true;   % verbosity, false by default
W0=zeros(size(X,2),size(Y,2));
param.loss='square';
fprintf('\nFISTA + Regression multi-task-graph \n');
param.regul='multi-task-graph';
param.lambda2=0.01;
tic
[W optim_info]=mexFistaGraph(Y,X,W0,graph,param);
toc
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

% Multi-Task Classification
fprintf('\nFISTA + Logistic + multi-task-graph \n');
param.regul='multi-task-graph';
param.lambda2=0.01;
param.loss='logistic';
Y=2*double(randn(100,size(Y,2)) > 0)-1;
tic
[W optim_info]=mexFistaGraph(Y,X,W0,graph,param);
toc
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
% Multi-Class + Multi-Task Regularization

param.verbose=false;
fprintf('\nFISTA + Multi-Class Logistic +multi-task-graph \n');
Y=double(ceil(5*rand(100,size(Y,2)))-1); 
param.loss='multi-logistic';
param.regul='multi-task-graph';
tic
nclasses=max(Y(:))+1;
W0=zeros(size(X,2),nclasses*size(Y,2));
[W optim_info]=mexFistaGraph(Y,X,W0,graph,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
% can be used of course with other regularization functions, intercept,...




