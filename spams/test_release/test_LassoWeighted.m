clear all;

fprintf('test Lasso weighted\n');
randn('seed',0);
% Data are generated
X=randn(64,10000);
X=X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);
D=randn(64,256);
D=D./repmat(sqrt(sum(D.^2)),[size(D,1) 1]);

% parameter of the optimization procedure are chosen
param.L=20; % not more than 20 non-zeros coefficients (default: min(size(D,1),size(D,2)))
param.lambda=0.15; % not more than 20 non-zeros coefficients
param.numThreads=8; % number of processors/cores to use; the default choice is -1
                    % and uses all the cores of the machine
param.mode=2;       % penalized formulation

W=rand(size(D,2),size(X,2));

tic
alpha=mexLassoWeighted(X,D,W,param);
t=toc;
toc

fprintf('%f signals processed per second\n',size(X,2)/t);
