clear all;
randn('seed',0);

fprintf('test mexL1L2BCD\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decomposition of a large number of groups 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=randn(64,100);
D=randn(64,200);
D=D./repmat(sqrt(sum(D.^2)),[size(D,1) 1]);
ind_groups=int32(0:10:size(X,2)-1); % indices of the first signals in each group

% parameter of the optimization procedure are chosen
param.itermax=100;
param.tol=1e-3;
param.mode=2; % penalty mode
param.lambda=0.15; % squared norm of the residual should be less than 0.1
param.numThreads=-1; % number of processors/cores to use; the default choice is -1
                    % and uses all the cores of the machine
tic
alpha0=zeros(size(D,2),size(X,2));
alpha=mexL1L2BCD(X,D,alpha0,ind_groups,param);
t=toc
fprintf('%f signals processed per second\n',size(X,2)/t);
