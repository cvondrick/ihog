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

clear param;
param.regul='l1';        % many other regularization functions are available
param.loss='square';     % only square and log are available
param.numThreads=1;    % uses all possible cores
param.normalized=false;  % if the columns of X have unit norm, set to true.
param.averaging_mode=0;  % no averaging, averaging was not really useful in experiments
param.weighting_mode=2;  % weights are in O(1/sqrt(n)) 
param.optimized_solver=true;
param.verbose=false;

% set grid of lambda
max_lambda=max(abs(X*y))/n;
tablambda=max_lambda*(2^(1/8)).^(0:-1:-50);  % order from large to small
param.lambda=tablambda;    % best to order from large to small
tabepochs=[1 2 3 5 10];  % in this script, we compare the results obtained when varying the number of passes over the data.

%% The problem which will be solved is
%%   min_beta  1/(2n) ||y-X' beta||_2^2 + lambda ||beta||_1
fprintf('EXPERIMENT: ALL LAMBDAS IN PARALLEL, no warm restart\n');
% we try different experiments when varying the number of epochs.
% the problems for different lambdas are solve INDEPENDENTLY in parallel
obj=[];
objav=[];
for ii=1:length(tabepochs)
   param.iters=tabepochs(ii)*n;   % one pass over the data
   fprintf('EXP WITH %d PASS\n',tabepochs(ii));
   nlambdas=length(param.lambda);
   Beta0=zeros(p,nlambdas);
   tic
   [Beta tmp]=mexStochasticProx(y,X,Beta0,param);
   toc
   yR=repmat(y,[1 nlambdas]);
   fprintf('Objective functions: \n');
   obj=[obj; 0.5*sum((yR-X'*Beta).^2)/n+param.lambda.*sum(abs(Beta))];
   obj
   if param.averaging_mode
      objav=[objav; 0.5*sum((yR-X'*tmp).^2)/n+param.lambda.*sum(abs(tmp))];
      objav
   end
   fprintf('Sparsity: \n');
   sum(Beta ~= 0)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXPERIMENT 2: L2 logistic regression 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('EXPERIMENT FOR LOGISTIC REGRESSION + l2\n');
y=sign(y);
param.regul='l2';        % many other regularization functions are available
param.loss='logistic';     % only square and log are available
obj=[];
objav=[];
for ii=1:length(tabepochs)
   param.iters=tabepochs(ii)*n;   % one pass over the data
   fprintf('EXP WITH %d PASS\n',tabepochs(ii));
   nlambdas=length(param.lambda);
   Beta0=zeros(p,nlambdas);
   tic
   [Beta tmp]=mexStochasticProx(y,X,Beta0,param);
   toc
   yR=repmat(y,[1 nlambdas]);
   fprintf('Objective functions: \n');
   obj=[obj;sum(log(1.0+exp(-yR .* (X'*Beta))))/n+0.5*param.lambda.*sum(abs(Beta.^2))];
   obj
   if param.averaging_mode
      objav=[objav; sum(log(1.0+exp(-yR .* (X'*tmp))))/n+0.5*param.lambda.*sum(abs(tmp))];
      objav
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXPERIMENT 3: L1 logistic regression 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('EXPERIMENT FOR LOGISTIC REGRESSION + l1\n');
y=sign(y);
param.regul='l1';        % many other regularization functions are available
param.loss='logistic';     % only square and log are available
obj=[];
objav=[];
for ii=1:length(tabepochs)
   param.iters=tabepochs(ii)*n;   % one pass over the data
   fprintf('EXP WITH %d PASS\n',tabepochs(ii));
   nlambdas=length(param.lambda);
   Beta0=zeros(p,nlambdas);
   tic
   [Beta tmp]=mexStochasticProx(y,X,Beta0,param);
   toc
   yR=repmat(y,[1 nlambdas]);
   fprintf('Objective functions: \n');
   obj=[obj; sum(log(1.0+exp(-yR .* (X'*Beta))))/n+param.lambda.*sum(abs(Beta))];
   obj
   if param.averaging_mode
      objav=[objav; sum(log(1.0+exp(-yR .* (X'*tmp))))/n+param.lambda.*sum(abs(tmp))];
      objav
   end
   fprintf('Sparsity: \n');
   sum(Beta ~= 0)
end


