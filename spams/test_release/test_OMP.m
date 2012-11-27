clear all;
randn('seed',0);

fprintf('test mexOMP');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decomposition of a large number of signals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=randn(64,100000);
D=randn(64,200);
D=D./repmat(sqrt(sum(D.^2)),[size(D,1) 1]);
% parameter of the optimization procedure are chosen
param.L=10; % not more than 10 non-zeros coefficients
param.eps=0.1; % squared norm of the residual should be less than 0.1
param.numThreads=-1; % number of processors/cores to use; the default choice is -1
                    % and uses all the cores of the machine
tic
alpha=mexOMP(X,D,param);
t=toc
fprintf('%f signals processed per second\n',size(X,2)/t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Regularization path of a single signal 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=randn(64,1);
D=randn(64,10);
param.L=5;
D=D./repmat(sqrt(sum(D.^2)),[size(D,1) 1]);
[alpha path]=mexOMP(X,D,param);
