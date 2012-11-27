clear all;
randn('seed',0);
% Data are generated
X=randn(20000,100);
X=X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);

% parameter of the optimization procedure are chosen
param.numThreads=-1; % number of processors/cores to use; the default choice is -1
                    % and uses all the cores of the machine

param.pos=0;                   
param.mode=1;       % projection on the l1 ball
param.thrs=2;
tic
X1=mexSparseProject(X,param);
t=toc;
toc
fprintf('%f signals of size %d projected per second\n',size(X,2)/t,size(X,1));
fprintf('Checking constraint: %f, %f\n',min(sum(abs(X1))),max(sum(abs(X1))));


param.mode=2;       % projection on the Elastic-Net
param.lambda1=0.15;

tic
X1=mexSparseProject(X,param);
t=toc;
toc
fprintf('%f signals of size %d projected per second\n',size(X,2)/t,size(X,1));
constraints=sum((X1.^2))+param.lambda1*sum(abs(X1));
fprintf('Checking constraint: %f, %f\n',min(constraints),max(constraints));

param.mode=6;       % projection on the FLSA
param.lambda1=0.7;
param.lambda2=0.7;
param.lambda3=1.0;

X=rand(2000,100);
X=X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);

tic
X1=mexSparseProject(X,param);
t=toc;
toc
fprintf('%f signals of size %d projected per second\n',size(X,2)/t,size(X,1));
constraints=0.5*param.lambda3*sum(X1.^2)+param.lambda1*sum(abs(X1))+param.lambda2*sum(abs(X1(2:end,:)-X1(1:end-1,:)));
fprintf('Checking constraint: %f, %f\n',mean(constraints),max(constraints));
fprintf('Projection is approximate (stops at a kink)\n',mean(constraints),max(constraints));

param.mode=6;       % projection on the FLSA
param.lambda1=0.7;
param.lambda2=0.7;
param.lambda3=1.0;

X=rand(2000,100);
X=X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);

tic
X1=mexSparseProject(X,param);
t=toc;
toc
fprintf('%f signals of size %d projected per second\n',size(X,2)/t,size(X,1));
constraints=0.5*param.lambda3*sum(X1.^2)+param.lambda1*sum(abs(X1))+param.lambda2*sum(abs(X1(2:end,:)-X1(1:end-1,:)));
fprintf('Checking constraint: %f, %f\n',mean(constraints),max(constraints));
fprintf('Projection is approximate (stops at a kink)\n',mean(constraints),max(constraints));
