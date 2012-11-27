format compact;
randn('seed',0);
param.numThreads=-1; % all cores (-1 by default)
param.verbose=true;   % verbosity, false by default
param.lambda=0.05; % regularization parameter
param.it0=10;      % frequency for duality gap computations
param.max_it=200; % maximum number of iterations
param.L0=0.1;
param.tol=1e-3;
param.intercept=false;
param.pos=false;

X=randn(100,200);
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
fprintf('\nFISTA + Regression l1\n');
param.loss='square';
param.regul='l1';
% param.regul='group-lasso-l2';
% param.size_group=10;
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

param.regul='l1';
fprintf('\nISTA + Regression l1\n');
param.ista=true;
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

fprintf('\nSubgradient Descent + Regression l1\n');
param.ista=false;
param.subgrad=true;
param.a=0.1;
param.b=1000; % arbitrary parameters
max_it=param.max_it;
it0=param.it0;
param.max_it=500;
param.it0=50;
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
param.subgrad=false;
param.max_it=max_it;
param.it0=it0;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Regression l2\n');
param.regul='l2';
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Regression l2 + sparse feature matrix\n');
param.regul='l2';
tic
[W optim_info]=mexFistaFlat(Y,sparse(X),W0,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));


fprintf('\nFISTA + Regression Elastic-Net\n');
param.regul='elastic-net';
param.lambda2=0.1;
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
fprintf('mean loss: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Group Lasso L2\n');
param.regul='group-lasso-l2';
param.size_group=2;  % all the groups are of size 2
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Group Lasso L2 with variable size of groups \n');
param.regul='group-lasso-l2';
param2=param;
param2.groups=int32(randi(5,1,size(X,2)));  % all the groups are of size 2
param2.lambda=10*param2.lambda;
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param2);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Trace Norm\n');
param.regul='trace-norm-vec';
param.size_group=5;
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Regression Fused-Lasso\n');
param.regul='fused-lasso';
param.lambda2=0.1;
param.lambda3=0.1; %
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
fprintf('mean loss: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Regression no regularization\n');
param.regul='none';
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
fprintf('mean loss: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Regression l1 with intercept \n');
param.intercept=true;
param.regul='l1';
tic
[W optim_info]=mexFistaFlat(Y,[X ones(size(X,1),1)],[W0; zeros(1,size(W0,2))],param); % adds a column of ones to X for the intercept
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Regression l1 with intercept+ non-negative \n');
param.pos=true;
param.regul='l1';
tic
[W optim_info]=mexFistaFlat(Y,[X ones(size(X,1),1)],[W0; zeros(1,size(W0,2))],param);
t=toc;
fprintf('mean loss: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),t,mean(optim_info(4,:)));
param.pos=false;
param.intercept=false;

fprintf('\nISTA + Regression l0\n');
param.regul='l0';
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
fprintf('mean loss: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),t,mean(optim_info(4,:)));

fprintf('\nOne classification experiment\n');
Y=2*double(randn(100,1) > 0)-1;
fprintf('\nFISTA + Logistic l1\n');
param.regul='l1';
param.loss='logistic';
param.lambda=0.01;
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
% can be used of course with other regularization functions, intercept,...
param.regul='l1';
param.loss='weighted-logistic';
param.lambda=0.01;
fprintf('\nFISTA + weighted Logistic l1 + sparse matrix\n');
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
% can be used of course with other regularization functions, intercept,...

param.loss='logistic';
fprintf('\nFISTA + Logistic l1 + sparse matrix\n');
tic
[W optim_info]=mexFistaFlat(Y,sparse(X),W0,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
% can be used of course with other regularization functions, intercept,...

% Multi-Class classification
Y=double(ceil(5*rand(100,1000))-1); 
param.loss='multi-logistic';
fprintf('\nFISTA + Multi-Class Logistic l1\n');
tic
nclasses=max(Y(:))+1;
W0=zeros(size(X,2),nclasses*size(Y,2));
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
% can be used of course with other regularization functions, intercept,...

% Multi-Task regression
Y=randn(100,100);
Y=Y-repmat(mean(Y),[size(Y,1) 1]);
Y=mexNormalize(Y);
param.compute_gram=false;
W0=zeros(size(X,2),size(Y,2));
param.loss='square';
fprintf('\nFISTA + Regression l1l2 \n');
param.regul='l1l2';
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
toc
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Regression l1linf \n');
param.regul='l1linf';
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
toc
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Regression l1l2 + l1 \n');
param.regul='l1l2+l1';
param.lambda2=0.1;
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
toc
fprintf('mean loss: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Regression l1linf + l1 \n');
param.regul='l1linf+l1';
param.lambda2=0.1;
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
toc
fprintf('mean loss: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Regression l1linf + row + columns \n');
param.regul='l1linf-row-column';
param.lambda2=0.1;
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
toc
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

% Multi-Task Classification
fprintf('\nFISTA + Logistic + l1l2 \n');
param.regul='l1l2';
param.loss='logistic';
Y=2*double(randn(100,100) > 0)-1;
tic
[W optim_info]=mexFistaFlat(Y,X,W0,param);
toc
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
% Multi-Class + Multi-Task Regularization

fprintf('\nFISTA + Multi-Class Logistic l1l2 \n');
Y=double(ceil(5*rand(100,1000))-1); 
param.loss='multi-logistic';
param.regul='l1l2';
tic
nclasses=max(Y(:))+1;
W0=zeros(size(X,2),nclasses*size(Y,2));
[W optim_info]=mexFistaFlat(Y,X,W0,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
% can be used of course with other regularization functions, intercept,...
