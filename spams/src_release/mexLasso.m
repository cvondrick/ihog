% 
% Usage:  [A [path]]=mexLasso(X,D,param);
%  or:    [A [path]]=mexLasso(X,Q,q,param);
%
% Name: mexLasso
%
% Description: mexLasso is an efficient implementation of the
%     homotopy-LARS algorithm for solving the Lasso. 
%     
%     if the function is called this way [A [path]]=mexLasso(X,D,param),
%     it aims at addressing the following problems
%     for all columns x of X, it computes one column alpha of A
%     that solves
%       1) when param.mode=0
%         min_{alpha} ||x-Dalpha||_2^2 s.t. ||alpha||_1 <= lambda
%       2) when param.mode=1
%         min_{alpha} ||alpha||_1 s.t. ||x-Dalpha||_2^2 <= lambda
%       3) when param.mode=2
%         min_{alpha} 0.5||x-Dalpha||_2^2 + lambda||alpha||_1 +0.5 lambda2||alpha||_2^2
%
%     if the function is called this way [A [path]]=mexLasso(X,Q,q,param),
%     it solves the above optimisation problem, when Q=D'D and q=D'x.
%
%     Possibly, when param.pos=true, it solves the previous problems
%     with positivity constraints on the vectors alpha
%
% Inputs: X:  double m x n matrix   (input signals)
%               m is the signal size
%               n is the number of signals to decompose
%         D:  double m x p matrix   (dictionary)
%               p is the number of elements in the dictionary
%         param: struct
%               param.lambda  (parameter)
%               param.lambda2  (optional parameter for solving the Elastic-Net)
%                              for mode=0 and mode=1, it adds a ridge on the Gram Matrix
%               param.L (optional), maximum number of steps of the homotopy algorithm (can
%                        be used as a stopping criterion)
%               param.pos (optional, adds non-negativity constraints on the
%                 coefficients, false by default)
%               param.mode (see above, by default: 2)
%               param.numThreads (optional, number of threads for exploiting
%                 multi-core / multi-cpus. By default, it takes the value -1,
%                 which automatically selects all the available CPUs/cores).
%               param.cholesky (optional, default false),  choose between Cholesky 
%                 implementation or one based on the matrix inversion Lemma
%               param.ols (optional, default false), perform an orthogonal projection
%                 before returning the solution.
%               param.max_length_path (optional) maximum length of the path, by default 4*p
%
% Output: A: double sparse p x n matrix (output coefficients)
%         path: optional,  returns the regularisation path for the first signal
%
% Note: this function admits a few experimental usages, which have not
%     been extensively tested:
%         - single precision setting (even though the output alpha is double 
%           precision)
%
% Author: Julien Mairal, 2009


