clear all;

fprintf('test mexCD\n');
randn('seed',0);
% Data are generated
X=randn(64,100);
X=X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);
D=randn(64,100);
D=D./repmat(sqrt(sum(D.^2)),[size(D,1) 1]);

% parameter of the optimization procedure are chosen
param.lambda=0.015; % not more than 20 non-zeros coefficients
param.numThreads=4; % number of processors/cores to use; the default choice is -1
param.mode=2;       % penalized formulation

tic
alpha=mexLasso(X,D,param);
t=toc;
toc
E=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
fprintf('%f signals processed per second for LARS\n',size(X,2)/t);
fprintf('Objective function for LARS: %g\n',E);

param.tol=0.001;
param.itermax=1000;
tic
alpha2=mexCD(X,D,sparse(size(alpha,1),size(alpha,2)),param);
t=toc;
toc

fprintf('%f signals processed per second for CD\n',size(X,2)/t);
E=mean(0.5*sum((X-D*alpha2).^2)+param.lambda*sum(abs(alpha2)));
fprintf('Objective function for CD: %g\n',E);
fprintf('With Random Design, CD can be much faster than LARS\n');

