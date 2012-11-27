U=randn(100,1000);

param.lambda=0.1; % regularization parameter
param.num_threads=-1; % all cores (-1 by default)
param.verbose=true;   % verbosity, false by default

% test l0
fprintf('\nprox l0\n');
param.regul='l0';
param.pos=false;       % false by default
param.intercept=false; % false by default
alpha=mexProximalFlat(U,param);

% test l1
fprintf('\nprox l1, intercept, positivity constraint\n');
param.regul='l1';
param.pos=true;       % can be used with all the other regularizations
param.intercept=true; % can be used with all the other regularizations     
alpha=mexProximalFlat(U,param);

% test l2
fprintf('\nprox squared-l2\n');
param.regul='l2';
param.pos=false;
param.intercept=false;
alpha=mexProximalFlat(U,param);

% test elastic-net
fprintf('\nprox elastic-net\n');
param.regul='elastic-net';
param.lambda2=0.1;
alpha=mexProximalFlat(U,param);

% test fused-lasso
fprintf('\nprox fused lasso\n');
param.regul='fused-lasso';
param.lambda2=0.1;
param.lambda3=0.1;
alpha=mexProximalFlat(U,param);

% test l1l2
fprintf('\nprox mixed norm l1/l2\n');
param.regul='l1l2';
alpha=mexProximalFlat(U,param);

% test l1linf
fprintf('\nprox mixed norm l1/linf\n');
param.regul='l1linf';
alpha=mexProximalFlat(U,param);

% test l1l2+l1
fprintf('\nprox mixed norm l1/l2 + l1\n');
param.regul='l1l2+l1';
param.lambda2=0.1;
alpha=mexProximalFlat(U,param);

% test l1linf+l1
fprintf('\nprox mixed norm l1/linf + l1\n');
param.regul='l1linf+l1';
param.lambda2=0.1;
alpha=mexProximalFlat(U,param);

% test l1linf-row-column
fprintf('\nprox mixed norm l1/linf on rows and columns\n');
param.regul='l1linf-row-column';
param.lambda2=0.1;
alpha=mexProximalFlat(U,param);

% test none
fprintf('\nprox no regularization\n');
param.regul='none';
alpha=mexProximalFlat(U,param);
