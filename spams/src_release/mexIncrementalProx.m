% 
% Usage:  [W [optim]]=mexIncrementalProx(y,X,W0,param);
%
% Name: mexIncrementalProx
%
% Description: mexIncremrentalProx implements the incremental algorithm MISO
% for composite optimization in a large scale setting.
%        X is a design matrix of size p x n
%        y is a vector of size n 
% WARNING, X is transposed compared to the functions mexFista*, and y is a vector
%        param.lambda is a vector that contains nlambda different values of the
%        regularization parameter (it can be a scalar, in that case nlambda=1)
%        W0: is a dense matrix of size p x nlambda   It is in fact ineffective
%        in the current release
%        W: is the output, dense matrix of size p x nlambda
%
%         - if param.loss='square' and param.regul corresponds to a regularization
%           function (currently 'l1' or 'l2'), the following problem is solved
%           w = argmin (1/n)sum_{i=1}^n 0.5(y_i- x_i^T w)^2 + lambda psi(w)
%         - if param.loss='logistic' and param.regul corresponds to a regularization
%           function (currently 'l1' or 'l2'), the following problem is solved
%           w = argmin (1/n)sum_{i=1}^n log(1+ exp(-y_ix_i^T w)) + lambda psi(w)
%           Note that here, the y_i's should be -1 or +1 
%          
%         The current release does not handle intercepts
%
% Inputs: y: double dense vector of size n
%         X: dense or sparse matrix of size p x n
%         W0: dense matrix of size p x nlambda
%         param: struct
%           param.loss (choice of loss)
%           param.regul (choice of regularization function)
%           param.lambda : vector of size nlambda
%           param.epochs: number of passes over the data
%           param.minibatches: size of the mini-batches: recommended value is 1
%              if X is dense, and min(n,ceil(1/density)) if X is sparse
%           param.warm_restart : (path-following strategy, very efficient when
%              providing an array of lambdas, false by default)
%           param.normalized : (optional, can be set to true if the x_i's have
%              unit l2-norm, false by default)
%           param.strategy (optional, 3 by default)
%                          0: no heuristics, slow  (only for comparison purposes)
%                          1: adjust the constant L on 5% of the data 
%                          2: adjust the constant L on 5% of the data + unstable heuristics (this strategy does not work)
%                          3: adjust the constant L on 5% of the data + stable heuristic (this is by far the best choice)
%           param.numThreads (optional, number of threads)
%           param.verbose (optional)
%           param.seed (optional, choice of the random seed)
%
% Output:  W:  double dense p x nlambda matrix
%          optim: optional, double dense 3 x nlambda matrix.
%              first row: values of the objective functions.
%              third row: computational time 
%
% Author: Julien Mairal, 2013


