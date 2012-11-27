% 
% Usage:  A=mexLassoWeighted(X,D,W,param);
%
% Name: mexLassoWeighted.  
%
% WARNING: This function has not been tested intensively
%
% Description: mexLassoWeighted is an efficient implementation of the
%     LARS algorithm for solving the weighted Lasso. It is optimized
%     for solving a large number of small or medium-sized 
%     decomposition problem (and not for a single large one).
%     It first computes the Gram matrix D'D and then perform
%     a Cholesky-based OMP of the input signals in parallel.
%     For all columns x of X, and w of W, it computes one column alpha of A
%     which is the solution of
%       1) when param.mode=0
%         min_{alpha} ||x-Dalpha||_2^2   s.t.  
%                                     ||diag(w)alpha||_1 <= lambda
%       2) when param.mode=1
%         min_{alpha} ||diag(w)alpha||_1  s.t.
%                                        ||x-Dalpha||_2^2 <= lambda
%       3) when param.mode=2
%         min_{alpha} 0.5||x-Dalpha||_2^2  +  
%                                         lambda||diag(w)alpha||_1 
%     Possibly, when param.pos=true, it solves the previous problems
%     with positivity constraints on the vectors alpha
%
% Inputs: X:  double m x n matrix   (input signals)
%               m is the signal size
%               n is the number of signals to decompose
%         D:  double m x p matrix   (dictionary)
%               p is the number of elements in the dictionary
%         W:  double p x n matrix   (weights)
%         param: struct
%            param.lambda  (parameter)
%            param.L (optional, maximum number of elements of each 
%            decomposition)
%            param.pos (optional, adds positivity constraints on the
%            coefficients, false by default)
%            param.mode (see above, by default: 2)
%            param.numThreads (optional, number of threads for exploiting
%            multi-core / multi-cpus. By default, it takes the value -1,
%            which automatically selects all the available CPUs/cores).
%
% Output:  A: double sparse p x n matrix (output coefficients)
%
% Note: this function admits a few experimental usages, which have not
%     been extensively tested:
%         - single precision setting (even though the output alpha is double 
%           precision)
%
% Author: Julien Mairal, 2009


