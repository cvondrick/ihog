format compact;
randn('seed',0);
param.numThreads=-1; % all cores (-1 by default)
param.lambda=0.05; % regularization parameter

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
fprintf('\nRidge Regression with conjugate gradient solver\n');
tic
[W]=mexRidgeRegression(Y,X,W0,param);
t=toc

