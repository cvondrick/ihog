% 
% Usage:   A=mexOMP(X,D,param);
% or       [A path]=mexOMP(X,D,param);
%
% Name: mexOMP
%
% Description: mexOMP is an efficient implementation of the
%     Orthogonal Matching Pursuit algorithm. It is optimized
%     for solving a large number of small or medium-sized 
%     decomposition problem (and not for a single large one).
%     It first computes the Gram matrix D'D and then perform
%     a Cholesky-based OMP of the input signals in parallel.
%     X=[x^1,...,x^n] is a matrix of signals, and it returns
%     a matrix A=[alpha^1,...,alpha^n] of coefficients.
%     
%     it addresses for all columns x of X, 
%         min_{alpha} ||alpha||_0  s.t  ||x-Dalpha||_2^2 <= eps
%         or
%         min_{alpha} ||x-Dalpha||_2^2  s.t. ||alpha||_0 <= L
%         or
%         min_{alpha} 0.5||x-Dalpha||_2^2 + lambda||alpha||_0 
%         
%
% Inputs: X:  double m x n matrix   (input signals)
%            m is the signal size
%            n is the number of signals to decompose
%         D:  double m x p matrix   (dictionary)
%            p is the number of elements in the dictionary
%            All the columns of D should have unit-norm !
%         param: struct
%            param.L (optional, maximum number of elements in each decomposition, 
%               min(m,p) by default)
%            param.eps (optional, threshold on the squared l2-norm of the residual,
%               0 by default
%            param.lambda (optional, penalty parameter, 0 by default
%            param.numThreads (optional, number of threads for exploiting
%            multi-core / multi-cpus. By default, it takes the value -1,
%            which automatically selects all the available CPUs/cores).
%
% Output: A: double sparse p x n matrix (output coefficients)
%         path (optional): double dense p x L matrix (regularization path of the first signal)
%
% Note: this function admits a few experimental usages, which have not
%     been extensively tested:
%      - single precision setting (even though the output alpha is double 
%        precision)
%      - Passing an int32 vector of length n to param.L provides
%        a different parameter L for each input signal x_i
%      - Passing a double vector of length n to param.eps and or param.lambda 
%        provides a different parameter eps (or lambda) for each input signal x_i
%
% Author: Julien Mairal, 2009


