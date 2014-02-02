clear all;
n=400000;
p=1000;
density=0.01;

% generate random data
format compact;
randn('seed',0);
rand('seed',0);
X=sprandn(p,n,density);
mean_nrm=mean(sqrt(sum(X.^2)));
X=X/mean_nrm;

% generate some true model
z=double(sign(full(sprandn(p,1,0.05))));  
y=X'*z;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXPERIMENT 1: Lasso
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('EXPERIMENT FOR LASSO\n');
nrm=sqrt(sum(y.^2));
y=y+0.01*nrm*randn(n,1);    % add noise to the model
nrm=sqrt(sum(y.^2));
y=y*(sqrt(n)/nrm);

% set optimization parameters
clear param;
param.regul='l1';        % many other regularization functions are available
param.loss='square';     % only square and log are available
param.numThreads=-1;    % uses all possible cores
param.normalized=false;  % if the columns of X have unit norm, set to true.
param.strategy=3;        % MISO with all heuristics
                         % 0: no heuristics, slow  (only for comparison purposes)
                         % 1: adjust the constant L on 5% of the data 
                         % 2: adjust the constant L on 5% of the data + unstable heuristics (this strategy does not work)
                         % 3: adjust the constant L on 5% of the data + stable heuristic (this is by far the best choice)
param.verbose=true;
param.minibatches=min(n,ceil(1/density));  % size of the minibatches, requires to store twice the size of X 

% set grid of lambda
max_lambda=max(abs(X*y))/n;
tablambda=max_lambda*(2^(1/8)).^(0:-1:-50);  % order from large to small
tabepochs=[1 2 3 5 10];  % in this script, we compare the results obtained when changing the number of passes on the data.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compare the solutions obtained with different epochs for one 
% value of lambda;
fprintf('EXPERIMENT: SINGLE LAMBDA\n');
param.lambda=tablambda(20); 
for ii=1:length(tabepochs)
   fprintf('EXP WITH %d PASS\n',tabepochs(ii));
   param.epochs=tabepochs(ii);   % one pass over the data
   Beta0=zeros(p,1);
   tic
   [Beta tmp]=mexIncrementalProx(y,X,Beta0,param);
   toc
   obj=tmp(1);
   fprintf('Objective functions: %f\n',obj);
   spar=sum(Beta ~= 0);
   fprintf('Sparsity: %d\n',spar);
end
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we do the same experiments, but we provide the array of lambda to 
% the function, and we compare two strategies, the second one implementing
% a warm restart with 
param.lambda=tablambda;
%% The problem which will be solved is
%%   min_beta  1/(2n) ||y-X' beta||_2^2 + lambda ||beta||_1
% the problems for different lambdas are solve INDEPENDENTLY in parallel
fprintf('EXPERIMENT: ALL LAMBDAS WITHOUT WARM RESTART\n');
param.warm_restart=false;
param.verbose=false;
obj=[];
spar=[];
for ii=1:length(tabepochs)
   param.epochs=tabepochs(ii);   % one pass over the data
   fprintf('EXP WITH %d PASS OVER THE DATA\n',tabepochs(ii));
   nlambdas=length(param.lambda);
   Beta0=zeros(p,nlambdas);
   tic
   [Beta tmp]=mexIncrementalProx(y,X,Beta0,param);
   toc
   fprintf('Objective functions: \n');
   obj=[obj; tmp(1,:)];
   obj
   fprintf('Sparsity: \n');
   spar=[spar; sum(Beta ~= 0)];
   spar
end

% the problems are here solved sequentially with warm restart
% this seems to be the prefered choice.
fprintf('EXPERIMENT: SEQUENTIAL LAMBDAS WITH WARM RESTART\n');
fprintf('A SINGLE CORE IS USED\n');
param.warm_restart=true;
param.num_threads=1;
obj=[];
spar=[];
for ii=1:length(tabepochs)
   param.epochs=tabepochs(ii);   % one pass over the data
   fprintf('EXP WITH %d PASS\n',tabepochs(ii));
   nlambdas=length(param.lambda);
   Beta0=zeros(p,nlambdas);
   tic
   [Beta tmp]=mexIncrementalProx(y,X,Beta0,param);
   toc
   fprintf('Objective functions: \n');
   obj=[obj; tmp(1,:)];
   obj
   fprintf('Sparsity: \n');
   spar=[spar; sum(Beta ~= 0)];
   spar
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXPERIMENT 2: L2 logistic regression 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('EXPERIMENT FOR LOGISTIC REGRESSION + l2\n');
y=sign(y);
param.regul='l2';        % many other regularization functions are available
param.loss='logistic';     % only square and log are available
param.num_threads=-1;    % uses all possible cores
%param.strategy=3;        % MISO with all heuristics
fprintf('EXPERIMENT: ALL LAMBDAS WITHOUT WARM RESTART\n');
param.warm_restart=false;
param.verbose=false;
obj=[];
spar=[];
for ii=1:length(tabepochs)
   param.epochs=tabepochs(ii);   % one pass over the data
   fprintf('EXP WITH %d PASS\n',tabepochs(ii));
   nlambdas=length(param.lambda);
   Beta0=zeros(p,nlambdas);
   tic
   [Beta tmp]=mexIncrementalProx(y,X,Beta0,param);
   toc
   yR=repmat(y,[1 nlambdas]);
   fprintf('Objective functions: \n');
   obj=[obj;tmp(1,:)];
   obj
end

fprintf('EXPERIMENT: SEQUENTIAL LAMBDAS WITH WARM RESTART\n');
fprintf('A SINGLE CORE IS USED\n');
param.warm_restart=true;
param.num_threads=1;
obj=[];
spar=[];
for ii=1:length(tabepochs)
   param.epochs=tabepochs(ii);   % one pass over the data
   fprintf('EXP WITH %d PASS\n',tabepochs(ii));
   nlambdas=length(param.lambda);
   Beta0=zeros(p,nlambdas);
   tic
   [Beta tmp]=mexIncrementalProx(y,X,Beta0,param);
   toc
   fprintf('Objective functions: \n');
   obj=[obj; tmp(1,:)];
   obj
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXPERIMENT 3: L1 logistic regression 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y=sign(y);
fprintf('EXPERIMENT FOR LOGISTIC REGRESSION + l1\n');
param.regul='l1';        % many other regularization functions are available
param.loss='logistic';     % only square and log are available
param.num_threads=-1;    % uses all possible cores
%param.strategy=3;        % MISO with all heuristics
fprintf('EXPERIMENT: ALL LAMBDAS WITHOUT WARM RESTART\n');
param.warm_restart=false;
param.verbose=false;
param.lambda=tablambda;
obj=[];
spar=[];
for ii=1:length(tabepochs)
   param.epochs=tabepochs(ii);   % one pass over the data
   fprintf('EXP WITH %d PASS\n',tabepochs(ii));
   nlambdas=length(param.lambda);
   Beta0=zeros(p,nlambdas);
   tic
   [Beta tmp]=mexIncrementalProx(y,X,Beta0,param);
   toc
   yR=repmat(y,[1 nlambdas]);
   fprintf('Objective functions: \n');
   obj=[obj;tmp(1,:)];
   obj
end

fprintf('EXPERIMENT: SEQUENTIAL LAMBDAS WITH WARM RESTART\n');
fprintf('A SINGLE CORE IS USED\n');
param.warm_restart=true;
param.num_threads=1;
obj=[];
spar=[];
for ii=1:length(tabepochs)
   param.epochs=tabepochs(ii);   % one pass over the data
   fprintf('EXP WITH %d PASS\n',tabepochs(ii));
   nlambdas=length(param.lambda);
   Beta0=zeros(p,nlambdas);
   tic
   [Beta tmp]=mexIncrementalProx(y,X,Beta0,param);
   toc
   fprintf('Objective functions: \n');
   obj=[obj; tmp(1,:)];
   obj
   fprintf('Sparsity: \n');
   spar=[spar; sum(Beta ~= 0)];
   spar
end




