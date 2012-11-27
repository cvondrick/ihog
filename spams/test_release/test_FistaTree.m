format compact;
param.num_threads=-1; % all cores (-1 by default)
param.verbose=false;   % verbosity, false by default
param.lambda=0.001; % regularization parameter
param.it0=10;      % frequency for duality gap computations
param.max_it=200; % maximum number of iterations
param.L0=0.1;
param.tol=1e-5;
param.intercept=false;
param.pos=false;

% Example 2 of tree structure
% tree structured groups:
% g1= {0 1 2 3 4 5 6 7 8 9}    root(g1) = { };
% g2= {0 1 2 3 4 5}            root(g2) = {0 1 2};
% g3= {3 4}                    root(g3) = {3 4};
% g4= {5}                      root(g4) = {5};
% g5= {6 7 8 9}                root(g5) = { };
% g6= {6 7}                    root(g6) = {6 7};
% g7= {8 9}                    root(g7) = {8};
% g8 = {9}                     root(g8) = {9};
tree.own_variables=  int32([0 0 3 5 6 6 8 9]);   % pointer to the first variable of each group
tree.N_own_variables=int32([0 3 2 1 0 2 1 1]); % number of "root" variables in each group
tree.eta_g=[1 1 1 2 2 2 2.5 2.5];       
tree.groups=sparse([0 0 0 0 0 0 0 0; ...
                    1 0 0 0 0 0 0 0; ...
                    0 1 0 0 0 0 0 0; ...
                    0 1 0 0 0 0 0 0; ...
                    1 0 0 0 0 0 0 0; ...
                    0 0 0 0 1 0 0 0; ...
                    0 0 0 0 1 0 0 0; ...
                    0 0 0 0 0 0 1 0]);  % first group should always be the root of the tree

X=randn(100,10);
X=X-repmat(mean(X),[size(X,1) 1]);
X=mexNormalize(X);
Y=randn(100,100);
Y=Y-repmat(mean(Y),[size(Y,1) 1]);
Y=mexNormalize(Y);
W0=zeros(size(X,2),size(Y,2));
% Regression experiments 
% 100 regression problems with the same design matrix X.
fprintf('\nVarious regression experiments\n');
param.compute_gram=true;
fprintf('\nFISTA + Regression tree-l2\n');
param.loss='square';
param.regul='tree-l2';
tic
[W optim_info]=mexFistaTree(Y,X,W0,tree,param);
t=toc;
fprintf('mean loss: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Regression tree-linf\n');
param.regul='tree-linf';
tic
[W optim_info]=mexFistaTree(Y,X,W0,tree,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

% works also with non tree-structured regularization. tree is ignored
fprintf('\nFISTA + Regression Fused-Lasso\n');
param.regul='fused-lasso';
param.lambda2=0.001;
param.lambda3=0.001; %
tic
[W optim_info]=mexFistaTree(Y,X,W0,tree,param);
t=toc;
fprintf('mean loss: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),t,mean(optim_info(4,:)));

fprintf('\nISTA + Regression tree-l0\n');
param.regul='tree-l0';
tic
[W optim_info]=mexFistaTree(Y,X,W0,tree,param);
t=toc;
fprintf('mean loss: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),t,mean(optim_info(4,:)));

fprintf('\nFISTA + Regression tree-l2 with intercept \n');
param.intercept=true;
param.regul='tree-l2';
tic
[W optim_info]=mexFistaTree(Y,[X ones(size(X,1),1)],[W0; zeros(1,size(W0,2))],tree,param);
t=toc;
fprintf('mean loss: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),t,mean(optim_info(4,:)));
param.intercept=false;

% Classification
fprintf('\nOne classification experiment\n');
Y=2*double(randn(100,size(Y,2)) > 0)-1;
fprintf('\nFISTA + Logistic + tree-linf\n');
param.regul='tree-linf';
param.loss='logistic';
param.lambda=0.001;
tic
[W optim_info]=mexFistaTree(Y,X,W0,tree,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
% can be used of course with other regularization functions, intercept,...

% Multi-Class classification
Y=double(ceil(5*rand(100,size(Y,2)))-1); 
param.loss='multi-logistic';
param.regul='tree-l2';
fprintf('\nFISTA + Multi-Class Logistic + tree-l2 \n');
tic
nclasses=max(Y(:))+1;
W0=zeros(size(X,2),nclasses*size(Y,2));
[W optim_info]=mexFistaTree(Y,X,W0,tree,param);
t=toc;
fprintf('mean loss: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),t,mean(optim_info(4,:)));
% can be used of course with other regularization functions, intercept,...

% Multi-Task regression
Y=randn(100,size(Y,2));
Y=Y-repmat(mean(Y),[size(Y,1) 1]);
Y=mexNormalize(Y);
param.compute_gram=false;
param.verbose=true;   % verbosity, false by default
W0=zeros(size(X,2),size(Y,2));
param.loss='square';
fprintf('\nFISTA + Regression multi-task-tree \n');
param.regul='multi-task-tree';
param.lambda2=0.001;
tic
[W optim_info]=mexFistaTree(Y,X,W0,tree,param);
toc
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

% Multi-Task Classification
fprintf('\nFISTA + Logistic + multi-task-tree \n');
param.regul='multi-task-tree';
param.lambda2=0.001;
param.loss='logistic';
Y=2*double(randn(100,size(Y,2)) > 0)-1;
tic
[W optim_info]=mexFistaTree(Y,X,W0,tree,param);
toc
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));

% Multi-Class + Multi-Task Regularization
param.verbose=false;
fprintf('\nFISTA + Multi-Class Logistic +multi-task-tree \n');
Y=double(ceil(5*rand(100,size(Y,2)))-1); 
param.loss='multi-logistic';
param.regul='multi-task-tree';
tic
nclasses=max(Y(:))+1;
W0=zeros(size(X,2),nclasses*size(Y,2));
[W optim_info]=mexFistaTree(Y,X,W0,tree,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
% can be used of course with other regularization functions, intercept,...

fprintf('\nFISTA + Multi-Class Logistic +multi-task-tree + sparse matrix \n');
tic
nclasses=max(Y(:))+1;
W0=zeros(size(X,2),nclasses*size(Y,2));
[W optim_info]=mexFistaTree(Y,sparse(X),W0,tree,param);
t=toc;
fprintf('mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f\n',mean(optim_info(1,:)),mean(optim_info(3,:)),t,mean(optim_info(4,:)));
