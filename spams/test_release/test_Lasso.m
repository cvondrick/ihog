clear all;

randn('seed',0);
fprintf('test mexLasso\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decomposition of a large number of signals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data are generated
X=randn(100,100000);
X=X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);
D=randn(100,200);
D=D./repmat(sqrt(sum(D.^2)),[size(D,1) 1]);

% parameter of the optimization procedure are chosen
%param.L=20; % not more than 20 non-zeros coefficients (default: min(size(D,1),size(D,2)))
param.lambda=0.15; % not more than 20 non-zeros coefficients
param.numThreads=-1; % number of processors/cores to use; the default choice is -1
                     % and uses all the cores of the machine
param.mode=2;        % penalized formulation

tic
alpha=mexLasso(X,D,param);
t=toc
fprintf('%f signals processed per second\n',size(X,2)/t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Regularization path of a single signal 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=randn(64,1);
D=randn(64,10);
D=D./repmat(sqrt(sum(D.^2)),[size(D,1) 1]);
param.lambda=0;
[alpha path]=mexLasso(X,D,param);
