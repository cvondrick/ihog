% 
% Usage:   A=mexLassoMask(X,D,B,param);
%
% Name: mexLassoMask
%
% Description: mexLasso is a variant of mexLasso that handles
%     binary masks. It aims at addressing the following problems
%     for all columns x of X, and beta of B, it computes one column alpha of A
%     that solves
%       1) when param.mode=0
%         min_{alpha} ||diag(beta)(x-Dalpha)||_2^2 s.t. ||alpha||_1 <= lambda
%       2) when param.mode=1
%         min_{alpha} ||alpha||_1 s.t. ||diag(beta)(x-Dalpha)||_2^2 
%                                                              <= lambda*||beta||_0/m
%       3) when param.mode=2
%         min_{alpha} 0.5||diag(beta)(x-Dalpha)||_2^2 +
%                                                 lambda*(||beta||_0/m)*||alpha||_1 +
%                                                 (lambda2/2)||alpha||_2^2
%     Possibly, when param.pos=true, it solves the previous problems
%     with positivity constraints on the vectors alpha
%
% Inputs: X:  double m x n matrix   (input signals)
%               m is the signal size
%               n is the number of signals to decompose
%         D:  double m x p matrix   (dictionary)
%               p is the number of elements in the dictionary
%         B:  boolean m x n matrix   (mask)
%               p is the number of elements in the dictionary
%         param: struct
%               param.lambda  (parameter)
%               param.L (optional, maximum number of elements of each 
%                 decomposition)
%               param.pos (optional, adds positivity constraints on the
%                 coefficients, false by default)
%               param.mode (see above, by default: 2)
%               param.lambda2  (optional parameter for solving the Elastic-Net)
%                              for mode=0 and mode=1, it adds a ridge on the Gram Matrix
%               param.numThreads (optional, number of threads for exploiting
%                 multi-core / multi-cpus. By default, it takes the value -1,
%                 which automatically selects all the available CPUs/cores).
%
% Output: A: double sparse p x n matrix (output coefficients)
%
% Note: this function admits a few experimental usages, which have not
%     been extensively tested:
%         - single precision setting (even though the output alpha is double 
%           precision)
%
% Author: Julien Mairal, 2010


