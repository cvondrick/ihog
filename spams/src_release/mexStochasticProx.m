% 
% Usage:  [W [W2]]=mexStochasticProx(y,X,W0,param);
%
% Name: mexStochasticProx
%
% Description: mexStochasticProx implements a proximal MM stochastic algorithm 
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
%           param.iters : number of iterations (n corresponds to one pass over the data)
%           param.minibatches: size of the mini-batches: recommended value is 1
%           param.normalized : (optional, can be set to true if the x_i's have
%              unit l2-norm, false by default)
%           param.weighting_mode : (optional, 1 by default),
%                0:  w_t = (t_0+1)/(t+t_0)
%                1:  w_t = ((t_0+1)/(t+t_0))^(0.75)
%                2:  w_t = ((t_0+1)/(t+t_0))^(5)
%           param.averaging_mode: (optional, false by default)
%                0: no averaging
%                1: first averaging mode for W2
%                2: second averaging mode for W2
%                WARNING: averaging can be very slow for sparse solutions
%           param.determineEta (optional, automatically choose the parameters of the
%             learning weights w_t, true by default) 
%           param.t0 (optional, set up t0 for weights w_t = ((1+t0)/(t+t0))^(alpha)
%           param.numThreads (optional, number of threads)
%           param.verbose (optional)
%           param.seed (optional, choice of the random seed)
%
% Output:  W:  double dense p x nlambda matrix (contains the solution without averaging)
%          W2:  double dense p x nlambda matrix (contains the solution with averaging)
%
% Author: Julien Mairal, 2013


